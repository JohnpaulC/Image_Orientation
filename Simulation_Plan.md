# Simulation

Orientation estimation method

1. Gradients Orientation Histogram
2. Matching points

The simulated objects

1. whole image
2. local region

## Method

#### Gradients Orientation Histogram (GOH)

```python
HoG_cal, angle_HoG
```

- bin numbers
- **threshold**
- Translation Value calculation

### Matching points (MP)

```python
match_angle, angle_cal
```

- Descriptor
- Final result calculation: median / mean

### Object Detection (OD)

```python
object_detection, object_capture
```

- detection threshold
- **objects registration**

## Plan

### Manual all angle

- Using local GOH, **errors analysis**
  - 0.5 --> bin_num = 720
-  Using local MP, **errors distribution** 

### Real experiment

- Resize the image
- Whole image estimation
- local region method