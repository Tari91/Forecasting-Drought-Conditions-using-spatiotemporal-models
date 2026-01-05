import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models

# ==========================================
# 1. SYNTHETIC DATA GENERATION
# ==========================================
def generate_synthetic_drought_data(n_samples=500, n_timesteps=12, grid_size=(16, 16)):
    data = np.random.uniform(0, 1, (n_samples, n_timesteps, *grid_size, 1))
    for i in range(n_samples):
        shift_x = np.random.randint(0, 2)
        shift_y = np.random.randint(0, 2)
        for t in range(1, n_timesteps):
            prev_frame = data[i, t-1]
            shifted_frame = np.roll(prev_frame, (shift_x, shift_y), axis=(0, 1))
            data[i, t] = 0.7 * prev_frame + 0.3 * shifted_frame + np.random.normal(0, 0.01, grid_size + (1,))
    data = np.clip(data, 0, 1)
    return data

# Parameters
N_SAMPLES = 800
TIMESTEPS = 10
GRID_SIZE = (16, 16)

full_data = generate_synthetic_drought_data(N_SAMPLES, TIMESTEPS, GRID_SIZE)
X = full_data[:, :-1, ...]
y = full_data[:, -1, ...]

# ==========================================
# 2. CONVLSTM MODEL ARCHITECTURE
# ==========================================
def build_spatiotemporal_model(input_shape):
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.ConvLSTM2D(filters=64, kernel_size=(3, 3), padding='same', return_sequences=True, activation='relu'),
        layers.BatchNormalization(),
        layers.ConvLSTM2D(filters=32, kernel_size=(3, 3), padding='same', return_sequences=False, activation='relu'),
        layers.BatchNormalization(),
        layers.Conv2D(filters=1, kernel_size=(3, 3), activation='sigmoid', padding='same')
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

input_dims = (TIMESTEPS - 1, GRID_SIZE[0], GRID_SIZE[1], 1)
model = build_spatiotemporal_model(input_dims)
model.summary()

# ==========================================
# 3. TRAINING
# ==========================================
print("\nStarting training...")
history = model.fit(X, y, batch_size=32, epochs=15, validation_split=0.2, verbose=1)

# ==========================================
# 4. VISUALIZATION OF RESULTS
# ==========================================
sample_idx = np.random.randint(0, 100)
test_input = X[sample_idx:sample_idx+1]
ground_truth = y[sample_idx]
prediction = model.predict(test_input)[0]

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
vmin, vmax = 0, 1
im0 = axes[0].imshow(test_input[0, -1, :, :, 0], cmap='RdYlBu', vmin=vmin, vmax=vmax)
axes[0].set_title(f"Observation (Month {TIMESTEPS-1})")
plt.colorbar(im0, ax=axes[0])

im1 = axes[1].imshow(ground_truth[:, :, 0], cmap='RdYlBu', vmin=vmin, vmax=vmax)
axes[1].set_title(f"Ground Truth (Month {TIMESTEPS})")
plt.colorbar(im1, ax=axes[1])

im2 = axes[2].imshow(prediction[:, :, 0], cmap='RdYlBu', vmin=vmin, vmax=vmax)
axes[2].set_title(f"Model Forecast (Month {TIMESTEPS})")
plt.colorbar(im2, ax=axes[2])

plt.tight_layout()
plt.show()
