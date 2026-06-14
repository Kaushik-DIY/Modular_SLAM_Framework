# Jetson Nano deployment — staged runbook (online ROS-fed SLAM)

Target: **Jetson Nano B, 4 GB, Ubuntu 18.04, ROS Melodic**. Runs the modular fusion
SLAM (`run_realtime.py`) **online**, fed live RGB-D + 2D-LiDAR + IMU over ROS. The
system Python stays untouched: an isolated **Miniforge Python 3.8** env is the SLAM
runtime; ROS keeps its Python 2.7 and feeds data through a small socket bridge.

> **Why staged:** the Python layer ports easily, but the native rebuilds carry real
> risk. We bring up `lidar` mode first (smallest native surface), prove the core on
> ARM, then add the visual modes. Honest expectation: budget a day or two of
> iterative build-debugging, not minutes. Biggest watch-item: **one OpenCV only**.

---

## M0 — Safety net (do FIRST)
```bash
sudo jetson_clocks                       # max clocks (perf + avoids thermal stalls)
# 6 GB swap (compiles + visual modes need headroom on 4 GB):
sudo fallocate -l 6G /var/swapfile && sudo chmod 600 /var/swapfile
sudo mkswap /var/swapfile && sudo swapon /var/swapfile
echo '/var/swapfile swap swap defaults 0 0' | sudo tee -a /etc/fstab
```
**Clone/snapshot the SD card now** (e.g. `dd` to a host) — any breakage becomes a
10-minute reflash, not a JetPack reinstall.

## A1 — Transfer (source only; NOT the venv)
On the dev box, make a trimmed bundle (the `.venv` is x86_64/py3.11 binaries — useless
on ARM; datasets/outputs aren't needed online):
```bash
cd /home/kaushik/slam_ws
tar --exclude='.git' --exclude='.venv*' --exclude='*/build' --exclude='datasets' \
    --exclude='*_outputs' --exclude='reference_audit' --exclude='*.so' \
    -czf /tmp/slam_ws_src.tgz .
# -> copy /tmp/slam_ws_src.tgz to USB -> on the Jetson:  mkdir -p ~/slam_ws && tar xzf slam_ws_src.tgz -C ~/slam_ws
```
(Keep `third_party/vocabs/ORBvoc.dbow3` — it's the 49 MB vocab the visual modes need.)

## A2 — System libs (additive; NO OpenCV via apt)
```bash
sudo apt update
sudo apt install -y build-essential libeigen3-dev libsuitesparse-dev \
                    freeglut3-dev libglu1-mesa-dev libgl1-mesa-dev git
# Do NOT `apt install libopencv-dev` — it shadows JetPack's CUDA OpenCV (ABI clash).
```

## A3 — Miniforge Python 3.8 env (isolated)
```bash
wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-aarch64.sh
bash Miniforge3-Linux-aarch64.sh -b -p ~/miniforge3 && source ~/miniforge3/bin/activate
conda create -y -n slam python=3.8 && conda activate slam
pip install cmake                                   # >=3.16 (bionic apt is 3.10)
pip install -r ~/slam_ws/requirements_jetson.txt    # numpy/scipy/yaml/matplotlib/pybind11 (wheels)
conda install -y -c conda-forge opencv              # ONE OpenCV for both py + the C++ link
python -c "import cv2, numpy, yaml; print('env ok', cv2.__version__)"
```
> If you prefer JetPack's system cv2 instead of conda's: skip `conda install opencv`,
> and `ln -s /usr/lib/python3.6/dist-packages/cv2*.so $(python -c 'import site;print(site.getsitepackages()[0])')/`
> — but then `find_package(OpenCV)` in the builds must point at the JetPack OpenCV. Conda OpenCV is simpler.

## M1 — `lidar` mode ONLINE (smallest viable; proves the native core)
Needs ONLY `fusion_core` (+ numpy/cv2/yaml). Build g2o → fusion_core:

1. **g2o static libs** (C++11; reuse `third_party/G2OPY_INSTALL_NOTES.md` for the patches):
   ```bash
   cd ~/slam_ws/third_party/g2opy && mkdir -p build && cd build
   cmake .. -DCMAKE_BUILD_TYPE=Release && make -j2     # produces lib/*.a + g2o/config.h
   ```
2. **fusion_core** (already C++17 in CMakeLists → stock gcc-7; no PPA):
   ```bash
   cd ~/slam_ws/third_party/fusion_core && mkdir -p build && cd build
   cmake .. -DCMAKE_BUILD_TYPE=RelWithDebInfo -DPython3_EXECUTABLE="$(which python)"
   make -j2 fusion_core                                # -j2: 4 GB can OOM at -j4
   cp fusion_core*.so "$(python -c 'import site;print(site.getsitepackages()[0])')/"
   python -c "import fusion_core as fc; print(fc.hello())"
   ```
3. **Online run** (two terminals):
   ```bash
   # terminal 1 (conda slam): start SLAM first — it binds the socket, then waits
   cd ~/slam_ws && MPLBACKEND=Agg python -m slam_core.fusion2.run_realtime \
       --source ros --mode lidar --dataset datasets/lab_hybrid   # --dataset = calibration dir (sensor_config.yaml) only
   # terminal 2 (ROS Melodic, system py2): start the bridge with YOUR topic names
   python ~/slam_ws/tools/ros/fusion_ros_bridge.py \
       --rgb /camera/color/image_raw --depth /camera/aligned_depth_to_color/image_raw \
       --scan /scan --imu /imu --socket /tmp/fusion_ros.sock
   ```
   Type `status` / `quit` on the SLAM terminal's stdin; the final corrected map is
   saved under `fusion2_outputs/realtime_*/`. **If the live map builds, ~80 % of the
   hard part is proven.** Watch RAM with `tegrastats` (target < ~3 GB).

   > A `--dataset` config dir must hold `sensor_config.yaml` (camera intrinsics +
   > the hector LiDAR profile name). Calibration is static config, not live data.

## M2 — add `icp` / `dbow` / the visual front-end (each optional)
```bash
# pydbow3 (dbow proposer + orb/orb_lidar visual FE):
cd ~/slam_ws && bash tools/build_pyslam_pydbow3_local.sh    # adapt the venv path inside to the conda env
# small_gicp (icp verifier; no aarch64 wheel -> source build):
pip install --no-binary :all: small_gicp     # or clone + cmake if that fails
```
Then run `--mode orb_lidar` / `--mode orb`. **RAM is the watch-item** here (visual
peaks ~1 GB); if OOM, the same runner stays up in `lidar` mode.

## M3 — all four modes, live-switchable
With the natives built, the cross-sensor live switch works online — type on the SLAM
stdin: `fe vo`, `fe lidar`, `fe s2m`, `verifier icp`, `proposer dbow`, `status`.

---

## Troubleshooting (the likely snags)
- **`import cv2` vs the extension segfaults / `undefined symbol`** → two OpenCVs. Use
  exactly one (conda-forge OR system); rebuild fusion_core/pydbow3 against that one.
- **g2o cmake fails on Eigen** → bionic ships Eigen 3.3; install 3.4 to a prefix and
  point `-DEIGEN3_INCLUDE_DIR`.
- **OOM during compile** → use `make -j1`/`-j2` and ensure the M0 swap is on.
- **Bridge "could not connect"** → start the SLAM side first (it binds the socket).
- **No keyframes / no map** → `rostopic hz` the four topics; check the bridge log shows
  `fwd rgbd/scan/imu` counts rising; widen `--slop` if rgb/depth never pair.
