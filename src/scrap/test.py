import jax
import jax.numpy as jnp
from jax import grad, jit, random
import optax

# Set random seed for reproducibility

def optimise(C, learning_rate=0.01,
             num_steps=5000, seed=0, verbose=True, x_min=-1.0, x_max=1.0):

    key = random.PRNGKey(seed)

    C = C.T

    # Define the problem dimensions
    M, N = C.shape

    # Initialize W with random values
    key, subkey = random.split(key)
    W = random.normal(subkey, (M, N))



    print("Initial W:")
    print(W)
    print("\nTarget C:")
    print(C.T)

    # Define the loss function
    def loss_fn(params):
        W, x = params
        return jnp.sum((W + x[:, jnp.newaxis] - C) ** 2)

    # Define the gradient function
    grad_fn = jit(grad(loss_fn))

    # Define the optimization step
    @jit
    def step(params, opt_state):
        grads = grad_fn(params)
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        W, x = params
        x = jnp.clip(x, x_min,x_max)
        params = W, x
        return params, opt_state


    # Initialize optimizer
    optimizer = optax.adam(learning_rate)

    # Initialize x
    x = jnp.zeros(M)

    # Pack W and x into a single parameter set
    params = (W, x)

    # Initialize optimizer state
    opt_state = optimizer.init(params)


    for i in range(num_steps):
        params, opt_state = step(params, opt_state)
        if i % 100000 == 0:
            W, x = params
            loss = loss_fn(params)
            print(f"Step {i}, Loss: {loss_fn(params):.6f}")
            #print(f"W:\n{W}")
            print(f"x: {x}\n")
    return W.T, x, loss


def main():
    C = jnp.array([
        [2, 3, 4],
        [6, 7, 8],
    [2, 3, 4],
        [6, 7, 8],
    [2, 3, 4],
        [6, 7, 8],
    [2, 3, 4],
        [6, 7, 8],
    [2, 3, 4],
        [6, 7, 8],
    [2, 3, 4],
        [6, 7, 8],
    [2, 3, 4],
        [6, 7, 8],

    ])
    import time
    start = time.time()
    W_final, x_final, loss = optimise(
        C, 0.01,500000,verbose=True, x_min=0.4, x_max=10.0
    )
    end = time.time()
    print(end - start, "Seconds")

    # Unpack final results


    # Print final results
    print("\nFinal W:")
    print(W_final)
    print("\nFinal x:")
    print(x_final)
    print("Final loss:", loss)

    # Verify the result
    result = W_final + x_final[jnp.newaxis, :]
    print("\nW + x:")
    print(result)
    print("\nC:")
    print(C)
    print("\nDifference (C - (W + x)):")
    print(C - result)

if __name__ == '__main__':
    main()