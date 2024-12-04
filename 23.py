import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button, Slider
from scipy.ndimage import gaussian_filter

# -----------------------------
# 1. Simulation Parameters
# -----------------------------

# Grid parameters
nx, ny = 100, 100          # Grid size representing neurons
dx, dy = 1.0, 1.0          # Spatial step size
dt = 0.1                   # Time step
c = 1.0                    # Wave speed (analogous to neuronal firing rate)
alpha = 0.05               # Nonlinearity coefficient (increased for visible patterns)
steps = 500                # Number of simulation steps

# Temperature parameters for phase transition
initial_temperature = 1.0  # Initial temperature (liquid)
final_temperature = 0.1    # Final temperature (ice)
temperature_decay = (initial_temperature - final_temperature) / steps

# Initialize temperature array
temperatures = initial_temperature - np.arange(steps) * temperature_decay

# -----------------------------
# 2. Initialize Membrane State
# -----------------------------

u = np.zeros((nx, ny))     # Amplitude (activation level)
v = np.zeros((nx, ny))     # Velocity or rate of change of amplitude
phi = np.zeros((nx, ny))   # Phase (state information)

# Initial condition: increased disturbance in the center
u[nx//2, ny//2] = 2.0  # Increased amplitude for visible dynamics
phi[nx//2, ny//2] = 0.0

# -----------------------------
# 3. Define Neuronal Network
# -----------------------------

# Define neuron locations (example)
neuron_locations = [(25, 25), (75, 75), (50, 50)]
# Define connections (source, target)
connections = [((25, 25), (50, 50)), ((50, 50), (75, 75))]

# -----------------------------
# 4. Define Vector-Based State Representation
# -----------------------------

def get_state_vector(u, phi):
    """
    Flattens the membrane's amplitude and phase into a single state vector.
    
    Parameters:
        u (np.ndarray): 2D array of membrane amplitudes.
        phi (np.ndarray): 2D array of membrane phases.
    
    Returns:
        np.ndarray: 1D state vector containing amplitude and phase information.
    """
    return np.concatenate((u.flatten(), phi.flatten()))

# -----------------------------
# 5. Define Neuronal Calculations
# -----------------------------

# Define neuronal network parameters
num_neurons = nx * ny * 2  # Number of neurons based on state vector size
synaptic_weights = np.random.randn(num_neurons, num_neurons) * 0.01  # Initialize synaptic weights

def sigmoid(x):
    """
    Applies the sigmoid activation function element-wise.
    """
    return 1 / (1 + np.exp(-x))

def neuronal_activation(state_vector):
    """
    Simulates neuronal activation based on the current state vector.
    
    Parameters:
        state_vector (np.ndarray): 1D state vector representing the membrane's state.
    
    Returns:
        np.ndarray: 1D array of neuronal activations.
    """
    # Simple linear transformation followed by sigmoid activation
    activations = sigmoid(np.dot(synaptic_weights, state_vector))
    return activations

def hebbian_learning(synaptic_weights, state_vector, activations, learning_rate=0.001):
    """
    Adjusts synaptic weights based on Hebbian learning rule.
    
    Parameters:
        synaptic_weights (np.ndarray): 2D array of synaptic weights.
        state_vector (np.ndarray): 1D state vector representing the membrane's state.
        activations (np.ndarray): 1D array of neuronal activations.
        learning_rate (float): Rate at which weights are updated.
    
    Returns:
        np.ndarray: Updated synaptic weights.
    """
    # Outer product of activations and state_vector
    delta_weights = learning_rate * np.outer(activations, state_vector)
    synaptic_weights += delta_weights
    return synaptic_weights

# -----------------------------
# 6. Define Update Functions
# -----------------------------

def update_membrane(u, v, phi, dx, dy, dt, c, beta, alpha, quantum_input, classical_input):
    """
    Updates the membrane's amplitude, velocity, and phase based on wave propagation, damping,
    nonlinearity, and external inputs.
    """
    laplacian = (
        -4 * u +
        np.roll(u, 1, axis=0) +
        np.roll(u, -1, axis=0) +
        np.roll(u, 1, axis=1) +
        np.roll(u, -1, axis=1)
    ) / (dx * dy)
    
    a = c**2 * laplacian - beta * v - alpha * (u**3) + quantum_input + classical_input
    v_new = v + a * dt
    u_new = u + v_new * dt
    phi_new = phi + (v_new * dt)
    
    return u_new, v_new, phi_new

def process_neurons(u, phi):
    """
    Processes neuronal activity by propagating activation from source neurons to target neurons.
    """
    for (source, target) in connections:
        sx, sy = source
        tx, ty = target
        if u[sx, sy] > 0.8:
            u[tx, ty] += 0.1
            phi[tx, ty] += np.pi / 4
    return u, phi

def autonomous_thought_cycles(frame, psi, cycle_interval=150):
    """
    Simulates autonomous thought cycles by inducing discharge events at random points.
    """
    if frame % cycle_interval == 0:
        # Select a random point for discharge
        ix, iy = np.random.randint(0, nx), np.random.randint(0, ny)
        psi[ix, iy] = 0.2  # Simulate discharge by reducing amplitude
    return psi

def apply_feedback(u, discharge_log, feedback_strength=0.1):
    """
    Applies feedback to the membrane based on recorded discharge events.
    """
    for event in discharge_log:
        frame, ix, iy = event
        # Apply a feedback influence around the discharge point
        radius = 5
        for x in range(max(ix - radius, 0), min(ix + radius, nx)):
            for y in range(max(iy - radius, 0), min(iy + radius, ny)):
                distance = np.sqrt((x - ix)**2 + (y - iy)**2)
                if distance < radius:
                    u[x, y] += feedback_strength * (1 - distance/radius)
    return u

def log_events(frame, psi, u_threshold=0.3, recharge_threshold=0.7):
    """
    Logs discharge and recharge events based on amplitude thresholds.
    """
    global discharge_log, recharge_log
    # Detect discharge events (significant drop in amplitude)
    discharge_indices = np.where(np.abs(psi) < u_threshold)
    for ix, iy in zip(*discharge_indices):
        discharge_log.append((frame, ix, iy))
    
    # Detect recharge events (significant rise in amplitude)
    recharge_indices = np.where(np.abs(psi) > recharge_threshold)
    for ix, iy in zip(*recharge_indices):
        recharge_log.append((frame, ix, iy))
    
    return discharge_log, recharge_log

# -----------------------------
# 7. Define Observation Handler
# -----------------------------

observation_active = False  # Flag to indicate if observation is active

def observe(event):
    """
    Toggles the observation flag when the Observe button is clicked.
    """
    global observation_active
    observation_active = not observation_active

# -----------------------------
# 8. Setup Visualization
# -----------------------------

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
ax1, ax2, ax3, ax4, ax5, ax6 = axes.flatten()

# Initialize plots
im1 = ax1.imshow(u, cmap='viridis', vmin=-1, vmax=2)
ax1.set_title('Membrane Amplitude')
ax1.axis('off')

im2 = ax2.imshow(phi, cmap='hsv', vmin=-np.pi, vmax=np.pi)
ax2.set_title('Membrane Phase')
ax2.axis('off')

im3 = ax3.imshow(np.zeros((nx, ny)), cmap='magma', vmin=0, vmax=1)
ax3.set_title('Energy Flow')
ax3.axis('off')

im4 = ax4.imshow(np.zeros((nx, ny)), cmap='cool', vmin=-2, vmax=2)
ax4.set_title('Information Density')
ax4.axis('off')

# Initialize temperature plot
ax5.set_xlim(0, steps)
ax5.set_ylim(final_temperature, initial_temperature)
ax5.set_title('Temperature Over Time')
ax5.set_xlabel('Time Step')
ax5.set_ylabel('Temperature')
temperature_line, = ax5.plot([], [], 'r-')

# Initialize observation area overlay
im6 = ax6.imshow(np.zeros((nx, ny)), cmap='Blues', alpha=0.5)
ax6.set_title('Observation Area')
ax6.axis('off')

# Add observation button
ax_observe = plt.axes([0.85, 0.02, 0.1, 0.04])
btn_observe = Button(ax_observe, 'Observe')
btn_observe.on_clicked(observe)

# Initialize markers for discharge and recharge
discharge_markers, = ax1.plot([], [], 'ro', markersize=3, label='Discharge')
recharge_markers, = ax1.plot([], [], 'go', markersize=3, label='Recharge')

# Add legends
ax1.legend(loc='upper right')

# Initialize text box to display current hash (Optional, can be removed if not needed)
hash_text = ax6.text(0.5, 0.95, '', transform=ax6.transAxes, 
                     fontsize=8, ha='center', va='top', 
                     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# -----------------------------
# 9. Define Calculation Functions
# -----------------------------

def calculate_energy_flow(u):
    """
    Calculates the energy flow across the membrane based on amplitude gradients.
    """
    grad_x = np.gradient(u, axis=0)
    grad_y = np.gradient(u, axis=1)
    energy_flow = np.sqrt(grad_x**2 + grad_y**2)
    # Normalize for visualization
    energy_flow = (energy_flow - energy_flow.min()) / (energy_flow.max() - energy_flow.min() + 1e-8)
    return energy_flow

def calculate_information_density(phi):
    """
    Calculates the information density based on phase coherence.
    """
    coherence = np.cos(phi - np.roll(phi, 1, axis=0)) + np.cos(phi - np.roll(phi, 1, axis=1))
    information_density = gaussian_filter(coherence, sigma=1)
    # Normalize for visualization
    information_density = (information_density - information_density.min()) / (information_density.max() - information_density.min() + 1e-8)
    return information_density

# -----------------------------
# 10. Define the Animation Update Function
# -----------------------------

# Initialize lists to track hashes and events
hash_history = []
discharge_log = []
recharge_log = []

def animate(frame):
    global u, v, phi, observation_active, discharge_log, recharge_log, hash_history, synaptic_weights
    
    # Update temperature
    current_temperature = temperatures[frame]
    
    # Adjust damping based on temperature (higher temperature -> lower damping)
    beta = 0.02 / current_temperature  # Adjusted damping
    
    # Quantum input: random noise proportional to temperature
    quantum_input = np.random.normal(0, 0.1 * current_temperature, (nx, ny))
    
    # Classical input: external stimuli
    classical_input = np.zeros((nx, ny))
    if frame == 100:
        classical_input[nx//3, ny//3] = 1.0
    if frame == 200:
        classical_input[2*nx//3, 2*ny//3] = 1.0
    if frame == 300:
        classical_input[nx//2, ny//2] = 1.0
    
    # Update membrane state
    u, v, phi = update_membrane(u, v, phi, dx, dy, dt, c, beta, alpha, quantum_input, classical_input)
    
    # Process neuronal activity
    u, phi = process_neurons(u, phi)
    
    # Compute the state vector
    state_vector = get_state_vector(u, phi)
    
    # Perform neuronal activation
    activations = neuronal_activation(state_vector)
    
    # Reshape activations back to 2D grid for influence
    activation_grid = activations[:nx*ny].reshape((nx, ny))  # Assuming first half corresponds to amplitudes
    
    # Influence membrane based on neuronal activations
    u += activation_grid * 0.05  # Scale influence as needed
    
    # Perform Hebbian learning (optional)
    synaptic_weights = hebbian_learning(synaptic_weights, state_vector, activations)
    
    # Apply observation-induced phase shift
    if observation_active:
        ox, oy = nx//2, ny//2
        collapse_radius = 10
        y, x = np.ogrid[:nx, :ny]
        distance = np.sqrt((x - ox)**2 + (y - oy)**2)
        collapse_mask = distance < collapse_radius
        dominant_phase = np.angle(u[ox, oy] + 1j*phi[ox, oy])
        phi[collapse_mask] = dominant_phase
        u[collapse_mask] *= 0.5
        observation_active = False
        
        # Highlight observation area
        im6.set_data(collapse_mask.astype(float))
    
    # Log discharge and recharge events
    discharge_log, recharge_log = log_events(frame, u)
    
    # Autonomous thought cycles (optional)
    u = autonomous_thought_cycles(frame, u)
    
    # Apply feedback based on discharge log (optional)
    u = apply_feedback(u, discharge_log)
    
    # Calculate energy flow and information density
    energy_flow = calculate_energy_flow(u)
    information_density = calculate_information_density(phi)
    
    # Update visualizations
    im1.set_data(u)
    im2.set_data(phi)
    im3.set_data(energy_flow)
    im4.set_data(information_density)
    
    # Update temperature plot
    temperature_line.set_data(np.arange(frame+1), temperatures[:frame+1])
    
    # Update discharge and recharge markers
    discharge_x = [iy for (_, ix, iy) in discharge_log]
    discharge_y = [ix for (_, ix, iy) in discharge_log]
    recharge_x = [iy for (_, ix, iy) in recharge_log]
    recharge_y = [ix for (_, ix, iy) in recharge_log]
    
    discharge_markers.set_data(discharge_x, discharge_y)
    recharge_markers.set_data(recharge_x, recharge_y)
    
    # Clear observation area if not active
    if not observation_active:
        im6.set_data(np.zeros((nx, ny)))
    
    # Update hash text (Optional, can be removed if not needed)
    # hash_display = f'Hash: {current_hash[:8]}...'  # Display first 8 chars for brevity
    # hash_text.set_text(hash_display)
    
    return [im1, im2, im3, im4, discharge_markers, recharge_markers, im6, temperature_line]

# -----------------------------
# 11. Run the Animation
# -----------------------------
if __name__ == "__main__":
    ani = FuncAnimation(fig, animate, frames=steps, interval=50, blit=True)
    plt.tight_layout()
    plt.show()
