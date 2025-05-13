import jax
import jax.numpy as jnp

x = jnp.ones((10000, 10000))
y = jnp.dot(x, x.T)
print(y.shape)

# Define a simple function
def simple_op(a, b):
  return jnp.add(a, b)

# JIT compile the function
jit_simple_op = jax.jit(simple_op)

# Example usage
a = jnp.array([1.0, 2.0, 3.0])
b = jnp.array([4.0, 5.0, 6.0])

result = jit_simple_op(a, b)
print("JIT compiled function result:", result)

# You can also use it as a decorator
@jax.jit
def simple_op_decorated(a, b):
    return jnp.multiply(a,b)

result_decorated = simple_op_decorated(a,b)
print("JIT decorated function result:", result_decorated)