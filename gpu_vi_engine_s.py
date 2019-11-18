# !/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Multiplies two square matrices together using a *single* block of threads and
global memory only. Each thread computes one element of the resulting matrix.
"""

import numpy as np
from pycuda import driver, compiler, gpuarray, tools
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
from collections import defaultdict
from pycuda.reduction import ReductionKernel
import numpy
import time
# -- initialize the device
import pycuda.autoinit


# Tran_prob_Matrix_gpu, Tran_index_Matrix_gpu, Reward_Matrix_gpu, value_vector_gpu,


def get_offsets_and_block_sizes(total_len, max_block_size):
    ret_list = []
    offset = 0
    remaining = total_len
    while remaining > 0:
        if (remaining < max_block_size):
            ret_list.append((offset, remaining))
            remaining -= remaining
            offset += remaining
        else:
            ret_list.append((offset, max_block_size))
            remaining -= max_block_size
            offset += max_block_size
    return ret_list


kernel_code_template = """
__global__ void MatrixMulKernel(float *TPM, float *TIM, float *RM,  float *V, float *offset,  float *VNEW, float *Error)
{
    // 2D Thread ID (assuming that only *one* block will be executed)
    int tx = %(MATRIX_SIZE)s * (blockDim.y*blockIdx.y + threadIdx.y) +  blockDim.x*blockIdx.x + threadIdx.x;
    if(%(ROW_COUNT)s > tx){


    // Pvalue is used to store the element of the matrix
    // that is computed by the thread
    float MaxPvalue = 0;

    // Each thread loads one row of M and one column of N,
    //   to produce one element of P. i for action
    // Aidx = action index, Sidx = state index, NSidx = next state index
    for (int i = 0; i < %(ACTION_COUNT)s; ++i) {
      float Pvalue = 0;
      for (int k = 0; k < %(COL_COUNT)s; ++k) {
          int Aidx = (int) i*%(COL_COUNT)s* %(ROW_COUNT)s ;
          int Sidx = (int) tx * %(COL_COUNT)s ;
          int NSidx = (int) k;
          float Tprob = TPM[Aidx + Sidx + NSidx];
          int Vidx = (int) TIM[Aidx + Sidx + NSidx];
          float NSvalue = V[Vidx];
          Pvalue +=  Tprob*RM[Aidx + Sidx + NSidx] + Tprob * %(LAMBDA)s * NSvalue;
      }
      if(i==0){
        MaxPvalue = Pvalue;
      }else{
        if(MaxPvalue < Pvalue){
        MaxPvalue = Pvalue;
        }
      }
    }

    // Write the matrix to device memory;
    // each thread writes one element
    Error[tx] = MaxPvalue - V[tx];
    VNEW[tx] = MaxPvalue;
    }
}
"""

import math as mth

def bellman_backup_step_gpu(Tran_prob_Matrix_gpu,
                            Tran_index_Matrix_gpu,
                            Reward_Matrix_gpu,
                            value_vector_gpu,
                            new_value_vector_gpu,
                            error_gpu,
                            LAMBDA=0.99,
                            max_threads=1000):
    ACTION_COUNT, ROW_COUNT, COL_COUNT = Tran_prob_Matrix_gpu.shape
    # get the kernel code from the template

    MATRIX_SIZE = mth.ceil(mth.sqrt(ROW_COUNT))
    BLOCK_SIZE = 32
    # by specifying the constant MATRIX_SIZE
    kernel_code = kernel_code_template % {
        'ROW_COUNT': ROW_COUNT,
        'COL_COUNT': COL_COUNT,
        'ACTION_COUNT': ACTION_COUNT,
        'MATRIX_SIZE': MATRIX_SIZE,
        'LAMBDA': LAMBDA
    }

    if MATRIX_SIZE % BLOCK_SIZE != 0:
        grid = (MATRIX_SIZE // BLOCK_SIZE + 1, MATRIX_SIZE // BLOCK_SIZE + 1, 1)
    else:
        grid = (MATRIX_SIZE // BLOCK_SIZE, MATRIX_SIZE // BLOCK_SIZE, 1)

    # compile the kernel code
    mod = compiler.SourceModule(kernel_code)

    # get the kernel function from the compiled module
    matrixmul = mod.get_function("MatrixMulKernel")

    offset_and_block_size = get_offsets_and_block_sizes(ROW_COUNT, max_threads)
    # for offset_, block_size_ in offset_and_block_size:
        # call the kernel on the card
        # print(Reward_Matrix_gpu.get()[0])

    offset_gpu = gpuarray.to_gpu(np.array([0]).astype('float32'))

    try:
        matrixmul(
            # inputs
            Tran_prob_Matrix_gpu, Tran_index_Matrix_gpu, Reward_Matrix_gpu, value_vector_gpu, offset_gpu,
            # output
            new_value_vector_gpu,error_gpu,
            grid=grid,
            # (only one) block of MATRIX_SIZE x MATRIX_SIZE threads
            block=(BLOCK_SIZE, BLOCK_SIZE, 1)
        )
    except:
        # print("BLOCK_SIZE,MATRIX_SIZE, grid", BLOCK_SIZE,MATRIX_SIZE, grid)
        import pdb; pdb.set_trace()


    # import pdb; pdb.set_trace();
    return new_value_vector_gpu


def VI_SOLVER_GPU(Tran_prob_Matrix,
                  Tran_index_Matrix,
                  Reward_Matrix,
                  Value_Vector,
                  no_of_backups=900,
                  epsilon=0.001,
                  verbose=False,
                  max_threads=1000):

    ACTION_COUNT, ROW_COUNT, COL_COUNT = Tran_prob_Matrix.shape

    # transfer host (CPU) memory to device (GPU) memory
    Tran_prob_Matrix_gpu = gpuarray.to_gpu(Tran_prob_Matrix)
    Tran_index_Matrix_gpu = gpuarray.to_gpu(Tran_index_Matrix)
    Reward_Matrix_gpu = gpuarray.to_gpu(Reward_Matrix)
    value_vector_gpu = gpuarray.to_gpu(Value_Vector)
    Error__gpu = gpuarray.empty((ROW_COUNT), np.float32)

    # create empty gpu array for the result (C = A * B)
    # import pdb; pdb.set_trace();

    # max_error_krnl = ReductionKernel(numpy.float32, neutral="0",
    #                                  reduce_expr="max(a,b)", map_expr="abs(x[i]-y[i])",
    #                                  arguments="float *x, float *y")

    for i in range(no_of_backups):
        try:
            new_value_vector_gpu = gpuarray.empty((ROW_COUNT), np.float32)
            bellman_backup_step_gpu(Tran_prob_Matrix_gpu, Tran_index_Matrix_gpu,Reward_Matrix_gpu,value_vector_gpu,new_value_vector_gpu,
                                    max_threads=max_threads, error_gpu = Error__gpu)
            if i+1 % 25 == 0:
                print("checkingggg for epsilng stop")
                max_error_gpu = pycuda.gpuarray.max(Error__gpu, stream = None) #((value_vector_gpu,new_value_vector_gpu)
                max_error = max_error_gpu.get()
                max_error_gpu.gpudata.free()
                if (verbose):
                    print(max_error)
                if max_error < epsilon:
                    break
            value_vector_gpu.gpudata.free()
            value_vector_gpu = new_value_vector_gpu
        except:
            print("HEYY SOMETHING WIERD HAPPENED HERE !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")


    # import pdb; pdb.set_trace();
    #
    value_vector_cpu = new_value_vector_gpu.get()
    Tran_prob_Matrix_gpu.gpudata.free()
    Tran_index_Matrix_gpu.gpudata.free()
    Reward_Matrix_gpu.gpudata.free()
    new_value_vector_gpu.gpudata.free()
    Error__gpu.gpudata.free()

    return value_vector_cpu


def get_tran_matrices_v2(td, rd, A, unknown_state_reward=-9999):
    print("Updated:unknown_state_reward: {} ".format(unknown_state_reward))
    td = cpy(td)
    rd = cpy(rd)
    td["unknown_state"] = {}
    rd["unknown_state"] = {}

    for a in A:
        td["unknown_state"][a] = {"unknown_state": 1}
        rd["unknown_state"][a] = unknown_state_reward

    max_ns_count = max([len(td[s][a]) for s in td for a in td[s]])
    tranMatrix = defaultdict(lambda: [])
    rewardMatrix = defaultdict(lambda: [])
    tranMatrixIndexes = defaultdict(lambda: [])
    state_to_index = {s: i for i, s in enumerate(td)}
    index_to_state = {i: s for i, s in enumerate(td)}


    for i, s in enumerate(td):
        for j, a in enumerate(A):
            next_state_probabilities = [td[s][a][ns] for ns in td[s][a]] if a in td[s] else []  # probabilities
            next_state_indexes = [state_to_index[ns] if ns in state_to_index else state_to_index["unknown_state"] for ns in
                                  td[s][a]] if a in td[s] else []  # indexes
            next_state_probabilities += [0] * (max_ns_count - len(next_state_probabilities)) if a in td[s] else [1/max_ns_count] *max_ns_count # padding
            next_state_indexes += [state_to_index["unknown_state"]] * (max_ns_count - len(next_state_indexes))  # padding
            assert len(next_state_indexes) == len(next_state_probabilities)

            tranMatrix[a].append(next_state_probabilities)
            rewardMatrix[a].append(
                [rd[s][a]] * len(next_state_probabilities) if a in rd[s] else [unknown_state_reward] * len(next_state_probabilities))
            tranMatrixIndexes[a].append(next_state_indexes)

    tranProbMatrix = np.array([tranMatrix[a] for a in tranMatrix])
    rewardValMatrix = np.array([rewardMatrix[a] for a in rewardMatrix])
    tranMatrixStateIndexes = np.array([tranMatrixIndexes[a] for a in tranMatrixIndexes])
    action_index_to_action = {i: a for i, a in enumerate(tranProbMatrix)}
    return tranProbMatrix, rewardValMatrix, action_index_to_action, tranMatrixStateIndexes, state_to_index, index_to_state


from copy import deepcopy as cpy


def gpu_value_iteration(S, A, reward_dict, tran_dict, seed_value=None, unknown_value=0, true_action_prob=0.9,
                        beta=0.99, epsilon=0.01, workers_num=4, verbose=True, max_threads=1000):
    import pycuda.autoinit
    # tran_dict, reward_dict = sanitize_transitions
    st = time.time()
    tranProbMatrix, rewardMatrix, action_index_to_action, tranMatrixStateIndexes, state_to_index, index_to_state = \
        get_tran_matrices_v2(tran_dict, reward_dict, A)

    et = time.time()

    print("tiem to get matris:{}".format(et - st))
    Tran_prob_Matrix, Tran_index_Matrix, Reward_Matrix = tranProbMatrix.astype(
        'float32'), tranMatrixStateIndexes.astype('float32'), rewardMatrix.astype('float32')
    ACTION_COUNT, number_of_states, number_of_connections = Tran_prob_Matrix.shape

    assert Tran_prob_Matrix.shape == Tran_index_Matrix.shape == Reward_Matrix.shape
    print(tranProbMatrix.shape, rewardMatrix.shape, tranMatrixStateIndexes.shape)

    Value_Vector = np.random.randint(1, size=(number_of_states, 1)).astype('float32').reshape(-1)
    # print(Value_Vector)
    ROW_COUNT = Tran_prob_Matrix.shape[1]
    COL_COUNT = Tran_prob_Matrix.shape[2]
    print(ACTION_COUNT, ROW_COUNT, COL_COUNT)
    assert Tran_prob_Matrix.shape == Tran_index_Matrix.shape

    ################################################################################################
    st = time.time()
    new_value_vector_cpu =  VI_SOLVER_GPU(Tran_prob_Matrix,
                                         Tran_index_Matrix,
                                         Reward_Matrix,
                                         Value_Vector,
                                         no_of_backups=1000,
                                         epsilon=epsilon,
                                         verbose=verbose,
                                         max_threads=max_threads)
    gpu_time = time.time() - st
    print(gpu_time)

    final_value_vector = new_value_vector_cpu #.get().reshape(-1)
    V_t = defaultdict(lambda: 0)
    for i, v in enumerate(final_value_vector):
        V_t[index_to_state[i]] = v

    # assert "end_state" in V_t
    # assert "end_state" in tran_dict
    # assert "end_state" in reward_dict
    assert "unknown_state" not in tran_dict
    # import pdb; pdb.set_trace();
    # print(V_t["end_state"])
    V_t = {k:V_t[k] for k in tran_dict}
    pi = get_pi_from_value(cpy(V_t), A, tran_dict, reward_dict, beta)
    #
    # assert "end_state" in pi
    # assert "end_state" in V_t
    assert "unknown_state" not in V_t
    return V_t, pi

def get_pi_from_value(V, list_of_actions, tran_dict, reward_dict, beta):
    v_max = {s: float('-inf') for s in V}
    pi = {}

    for s in tran_dict:
        for a in tran_dict[s]:
            expected_val = 0
            for ns in tran_dict[s][a]:
                try:
                    expected_val += tran_dict[s][a][ns] * V[ns]
                except:
                    expected_val += tran_dict[s][a][ns] * 0
            expect_s_val = reward_dict[s][a] + beta * expected_val
            if expect_s_val > v_max[s]:
                v_max[s] = expect_s_val
                pi[s] = a

            if (s == "end_state"):
                print(expect_s_val,reward_dict[s][a] ,tran_dict[s][a][ns],  V[ns], v_max[s])
    return pi
