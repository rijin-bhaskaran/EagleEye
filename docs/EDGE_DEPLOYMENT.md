# Edge deployment plan: Raspberry Pi now, AI accelerator later (ONNX route)

**Status: a plan. Nothing below has been run on a Raspberry Pi or an accelerator yet.** The only measured number so far is on an x86 laptop:
the passenger loop ran at 3.5 FPS (294 ms per frame) with both monitors running, against a target of 10-12 FPS.

## Why ONNX
ONNX is the common starting point for every accelerator toolchain, and it also runs on the Pi's CPU through ONNX Runtime. So one export serves both
phases: the Pi today, and the accelerator later. We avoid optimising for the Pi CPU alone (for example NCNN), because the accelerator will replace that path.

## Principles
1. **Train and export on a PC/GPU machine, never on the Pi.** The Pi only runs the exported file.
2. **The registry gate decides.** Every exported or quantised model is scored on the same frozen test split as the PyTorch champion
   (`python scripts/qa_gate.py`, `python -m passenger.registry`). A speed-up that loses recall or classification F1 beyond tolerance is rejected.
3. **Keep the FP32 champion** as the rollback. Never delete it when a faster model is promoted.
4. **The raw data is the calibration data.** Quantised (INT8) accelerators need a few hundred representative images to calibrate. They come from
   `raw/` and `processed/`, which is another reason the originals are never discarded.

## Steps

### Step 1: ONNX on the Pi CPU (can start as soon as there is a Pi)
1. On the training machine: export the champion, for example `YOLO("champion.pt").export(format="onnx", imgsz=640, simplify=True)`.
   This needs the `onnx` package there (not installed in this repo's venv yet). Use a fixed input size; accelerators want static shapes.
2. Ultralytics loads `.onnx` directly through ONNX Runtime, so **`PersonDetector` is expected to work unchanged**: set
   `passenger.detector.model: champion.onnx` in `config.yaml`. Verify this claim on the Pi before relying on it.
3. Install `onnxruntime` (aarch64 wheel) on the Pi.
4. **Parity check** on the dev machine first: run `python -m passenger.evaluate --split test` with the `.onnx` file and compare with the `.pt` scores.
   They should match within tolerance. Register it as its own model version.
5. **Benchmark on the Pi**: `python scripts/bench_passenger_overhead.py` (our code), then `python run_all.py ...` and read FPS / latency on the dashboard
   with the classifier on and off, at `imgsz` 640 and 480. Record the numbers in WORKFLOW.md.

### Step 2: choose the accelerator (decision point)
The accelerator decides the model format and toolchain. To confirm when chosen (check the vendor's current docs; these are expectations, not verified here):

| If you choose | Likely path | Watch out for |
|---|---|---|
| **Hailo** (for example the Raspberry Pi AI Kit, Hailo-8L) | ONNX -> vendor compiler on an x86 Linux host -> compiled model file; INT8 with calibration images | Vendor model zoo may already offer a compiled YOLOv8 person detector. Supported operators and input sizes are fixed at compile time. |
| **Coral Edge TPU** | ONNX -> TensorFlow Lite INT8 -> edge compiler | Conversion can be lossy for newer YOLO heads; usually the lowest accuracy of the options. |
| **Another NPU** | ONNX -> vendor toolchain | Check YOLOv8 operator support before buying. |

### Step 3: accelerator integration (work items, not yet built)
- **A second detector backend** with the same `detect(frame, track) -> List[Detection]` contract, chosen by a config key
  (for example `passenger.detector.backend`). Pre- and post-processing (resize, decoding, NMS) may move out of Ultralytics.
- **Tracking**: Ultralytics' ByteTrack comes with its own pipeline. With a vendor runtime we need a standalone tracker feeding `Detection.track_id`.
- **Quantisation check**: compile with calibration images drawn from our data, then run the gate. INT8 usually costs a little accuracy; the gate measures how much.

### Step 4: what an accelerator does *not* fix
- The **face finder (MediaPipe)** and the **age/gender nets (OpenCV DNN)** run on the Pi CPU. About 50 ms per classified person on the x86 laptop, three per frame
  at most, so up to about 150 ms of CPU per frame on a faster machine and more on a Pi. Options: lower `passenger.classifier.budget_per_frame` to 1,
  classify only new passengers, or later convert the classifier (a retrained, smaller face-attribute net) to the accelerator too.
- The **driver monitor** (MediaPipe face mesh, YOLO for phone detection, hands) keeps using the same CPU. Measure both monitors running together.
- **Dashboard assets** load from public CDNs: bundle them for offline use.

## Acceptance criteria for "the Pi is ready"
- Passenger loop sustains **10 FPS or better with both monitors running** for at least 10 minutes (read from the dashboard).
- **No thermal throttling** (check `vcgencmd get_throttled` and temperature during the run; add a heatsink/fan if needed).
- Quality gate passes against the FP32 champion on the frozen test split (no regression beyond the registry tolerance).
- Driver-monitor FPS and alert latency are not worse than before the passenger monitor was added.
- If the target FPS is not met: lower `imgsz`, the classifier budget or the target FPS in `config.yaml`, in that order, and record the trade-off.

## What to tell me when you decide
Which Raspberry Pi model, which accelerator, and whether the driver monitor will share the same board. Then Step 3 can be built against that specific toolchain.
