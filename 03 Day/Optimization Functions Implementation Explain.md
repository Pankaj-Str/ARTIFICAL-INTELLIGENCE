# Optimization Functions Implementation

###  Break down this code section by section to help you understand it better:

1. First, let's look at the Optimizers class setup:
```python
class Optimizers:
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate
```
This creates a class with a learning rate parameter that controls how big steps we take during optimization.

1. Each optimization method:

a) Standard Gradient Descent:
```python
def gradient_descent(self, params, gradients):
    return params - self.learning_rate * gradients
```
- Simplest form of optimization
- Takes current parameters and subtracts learning_rate × gradients
- Like walking downhill by taking steps in the steepest direction

b) Momentum:
```python
def momentum(self, params, gradients, velocity, momentum=0.9):
    velocity = momentum * velocity + self.learning_rate * gradients
    return params - velocity, velocity
```
- Adds "momentum" like a rolling ball
- Keeps track of previous updates through velocity
- momentum=0.9 means it keeps 90% of previous velocity
- Helps overcome small obstacles and speeds up convergence

c) RMSprop:
```python
def rmsprop(self, params, gradients, cache, decay_rate=0.9, epsilon=1e-8):
    cache = decay_rate * cache + (1 - decay_rate) * np.square(gradients)
    update = self.learning_rate * gradients / (np.sqrt(cache) + epsilon)
    return params - update, cache
```
- Adapts learning rate for each parameter
- Keeps track of squared gradients in cache
- Divides learning rate by square root of cache
- Handles different scales of parameters better

d) Adam:
```python
def adam(self, params, gradients, moment, velocity, t, 
        beta1=0.9, beta2=0.999, epsilon=1e-8):
    moment = beta1 * moment + (1 - beta1) * gradients
    velocity = beta2 * velocity + (1 - beta2) * np.square(gradients)
    
    # Bias correction
    moment_corrected = moment / (1 - beta1**t)
    velocity_corrected = velocity / (1 - beta2**t)
    
    update = self.learning_rate * moment_corrected / (np.sqrt(velocity_corrected) + epsilon)
    return params - update, moment, velocity
```
- Combines benefits of momentum and RMSprop
- Keeps track of both velocity (momentum) and squared gradients
- Includes bias correction for better early iterations
- Generally performs best in practice

3. Problem Setup:
```python
def objective_function(x, y):
    return x**2 + y**2

def gradient_function(x, y):
    return np.array([2*x, 2*y])

x = np.linspace(-2, 2, 100)
y = np.linspace(-2, 2, 100)
X, Y = np.meshgrid(x, y)
Z = objective_function(X, Y)
```
- Creates a simple bowl-shaped function to optimize
- Minimum is at (0,0)
- Sets up grid for visualization

4. Optimization Loop:
```python
for t in range(n_iterations):
    # For each optimizer (GD, Momentum, RMSprop, Adam):
    grad = gradient_function(*current_points['GD'])  # Calculate gradient
    current_points['GD'] = optimizer.gradient_descent(current_points['GD'], grad)  # Update position
    paths['GD'].append(current_points['GD'].copy())  # Store path for plotting
```
- Runs each optimizer for n_iterations
- Tracks path taken by each optimizer
- Updates current position using respective optimization methods

5. Visualization:
```python
plt.contour(X, Y, Z, levels=np.logspace(-2, 2, 20))  # Draw contour of function
colors = {'GD': 'red', 'Momentum': 'blue', 'RMSprop': 'green', 'Adam': 'purple'}
for method, path in paths.items():
    plt.plot(path[:, 0], path[:, 1], 'o-', label=method, color=colors[method])
```
- Shows contour of the function (like a topographic map)
- Plots path taken by each optimizer in different colors
- Helps visualize how each method approaches the minimum

Key differences between optimizers:
1. Gradient Descent: Takes direct steps down
2. Momentum: Like a ball rolling downhill
3. RMSprop: Takes bigger steps in flat areas, smaller steps in steep areas
4. Adam: Combines momentum's speed with RMSprop's adaptivity

