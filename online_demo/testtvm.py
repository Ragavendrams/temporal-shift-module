import logging
import sys
import numpy as np
import tvm
from tvm import te, autotvm

@autotvm.template('matmul')
def matmul(N, L, M, dtype):
    a = te.placeholder((N, L), name='A', dtype=dtype)
    b = te.placeholder((L, M), name='B', dtype=dtype)

    k = te.reduce_axis((0, L), name='k')
    c = te.compute((N, M), lambda i, j: te.sum(a[i, k] * b[k, j], axis=k), name='C')
    s = te.create_schedule(c.op)

    # schedule
    y, x = s[c].op.axis
    k = s[c].op.reduce_axis[0]

    ##### define space begin #####
    cfg = autotvm.get_config()
    cfg.define_split("tile_y", y, num_outputs=2)
    cfg.define_split("tile_x", x, num_outputs=2)
    ##### define space end #####

    # schedule according to config
    yo, yi = cfg["tile_y"].apply(s, c, y)
    xo, xi = cfg["tile_x"].apply(s, c, x)

    s[c].reorder(yo, xo, k, yi, xi)

    return s, [a, b, c]

N, L, M = 512, 512, 512
task = autotvm.task.create('matmul', args=(N, L, M, 'float32'), target='llvm')
print(task.config_space)

# logging config (for printing tuning log to the screen)
logging.getLogger('autotvm').setLevel(logging.DEBUG)
logging.getLogger('autotvm').addHandler(logging.StreamHandler(sys.stdout))

# There are two steps for measuring a config: build and run.
# By default, we use all CPU cores to compile program. Then measure them sequentially.
# We measure 5 times and take average to reduce variance.
measure_option = autotvm.measure_option(
    builder='local',
    # runner=autotvm.LocalRunner(number=5)
    runner=autotvm.LocalRunner(number=10, repeat=3, timeout=4, min_repeat_ms=150))

# Begin tuning with RandomTuner, log records to file `matmul.log`
# You can use alternatives like XGBTuner.
tuner = autotvm.tuner.XGBTuner(task,loss_type='rank')
tuner.tune(n_trial=100,
           measure_option=measure_option,
           callbacks=[autotvm.callback.log_to_file('matmul.log')])


with autotvm.apply_history_best('matmul.log'):
    with tvm.target.create('llvm'):
        s, arg_bufs = matmul(N, L, M, 'float32')
        func = tvm.build(s, arg_bufs)

# check correctness
a_np = np.random.uniform(size=(N, L)).astype(np.float32)
b_np = np.random.uniform(size=(L, M)).astype(np.float32)
c_np = a_np.dot(b_np)

c_tvm = tvm.nd.empty(c_np.shape)
func(tvm.nd.array(a_np), tvm.nd.array(b_np), c_tvm)

tvm.testing.assert_allclose(c_np, c_tvm.asnumpy(), rtol=1e-2)
print(c_tvm.asnumpy(),c_np)