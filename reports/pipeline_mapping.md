# LogiEdge 10-Stage Edge ML Pipeline Mapping

## 1. Data Collection

The LogiEdge simulator collects synchronized cold-chain observations for each
truck: temperature at 1 Hz, vibration RMS at 0.5 Hz, and timestamped door
events, publishing them through the local Mosquitto broker. It generates
Normal, temperature-drift, vibration-anomaly, and combined-anomaly scenarios
without relying on cellular connectivity.

## 2. Labelling

Labels are assigned automatically from the simulator mode: `none` becomes
Class 0 Normal, `temp_drift` becomes Class 1 Warning, and `combined` becomes
Class 2 Critical. This gives each generated 30-second feature window a
deterministic ground-truth label associated with the anomaly active during
that simulation run.

## 3. Training

LogiEdge trains a TensorFlow multilayer perceptron using the six normalized
fused features, with hidden layers of 32 and 16 ReLU units and a three-neuron
softmax output. Training uses the generated Normal, Warning, and Critical
windows, fixed normalization statistics from ten minutes of clean Normal
data, class weighting, early stopping, and a reproducible seed.

## 4. Validation

The trained model is evaluated on a reproducible held-out 20% validation
split and achieved 100% accuracy, with all 24 Normal, 18 Warning, and 18
Critical windows classified correctly. The correct-statistics and
3-sigma-shifted-statistics experiments both produced 100% accuracy, giving a
measured accuracy change of 0.00 percentage points on the controlled
synthetic validation data.

## 5. Optimisation

The current M1 baseline is the reference FP32 model, while subsequent
optimisation will produce the required full-INT8 post-training-quantized
model and the structured-pruned-plus-INT8 variant. These variants will be
compared for latency, p95 latency, file size, validation accuracy, energy per
inference, and Critical-class recall before selecting the truck-deployment
model.

## 6. Conversion

The validated Keras model is converted to the TFLite FlatBuffer format for
edge execution, producing `inference/model.tflite` with input shape `[1, 6]`
and output shape `[1, 3]`. The 5,608-byte FP32 TFLite artifact was checked
against the held-out validation set and packaged into the Docker inference
service.

## 7. Registration

Git and GitHub currently provide lightweight model registration and lineage
by versioning the simulator, preprocessing code, `training_stats.npy`,
dataset metadata, training script, Keras model, TFLite model, conversion
metadata, and evaluation results in one repository. Model identity is
additionally recorded through file hashes, including the SHA-256 values used
to distinguish the baseline and OTA-updated TFLite artifacts.

## 8. CI/CD

The current deployment pipeline builds an inference image from
`python:3.11-slim`, installs dependencies before copying the TFLite model,
validates the image, and supports runtime model selection through
`MODEL_PATH`. Docker layer caching was demonstrated by changing only
`model.tflite`: the earlier dependency and application layers were reused
while the model-copy layer was rebuilt; Ansible-based fleet deployment will
extend this stage in Task E2.

## 9. Inference

On each truck, the Dockerized service subscribes to
`logibridge/trucks/{truck_id}/sensors/combined`, applies the five-sample
filters, creates 30-second windows at ten-second steps, extracts and
normalizes the six fused features, and runs TFLite inference locally. It
publishes the predicted Normal, Warning, or Critical class, confidence,
probabilities, and latency to
`logibridge/trucks/{truck_id}/inference`.

## 10. Monitor and Update

Task E1 will monitor the rolling distribution of output confidence using
Population Stability Index and raise a drift alert when PSI exceeds 0.25;
detected degradation will provide evidence for collecting new labelled data
and starting another training cycle. Updated models can then be validated
and distributed through the Docker model layer and the selected OTA strategy,
completing the feedback loop from deployed trucks back to Data Collection.

## Circular Lifecycle

The pipeline is circular rather than ending at deployment. Inference
confidence and drift evidence from the deployed trucks feed the monitoring
stage; confirmed degradation leads to new data collection, labelling,
retraining, validation, conversion, and controlled fleet redeployment.
