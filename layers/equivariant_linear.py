# import tensorflow as tf
import torch
import numpy as np

class equi_2_to_2(torch.nn.Module):
    """equivariant nn layer."""

    def __init__(self, input_depth, output_depth, device):
        super(equi_2_to_2, self).__init__()
        self.basis_dimension = 15
        self.device = device
#         self.coeffs_values = torch.matmul(torch.randn(size=(input_depth, output_depth, self.basis_dimension), dtype=torch.float32), torch.sqrt(2. / (input_depth + output_depth).type(torch.FloatTensor)))
        self.coeffs_values = torch.mul(torch.randn(size=(input_depth, output_depth, self.basis_dimension), dtype=torch.float32), torch.sqrt(torch.tensor([2.]) / (input_depth + output_depth)))#.cuda()

        self.coeffs = torch.nn.Parameter(self.coeffs_values, requires_grad=False)
        self.diag_bias = torch.nn.Parameter(torch.zeros((1, output_depth, 1, 1), dtype=torch.float32), requires_grad=False)
        self.all_bias = torch.nn.Parameter(torch.zeros((1, output_depth, 1, 1), dtype=torch.float32), requires_grad=False)
        
    def ops_2_to_2(self, inputs, dim, normalization='inf', normalization_val=1.0):  # N x D x m x m
#         print(f'input shape : {inputs.shape}')
        diag_part = torch.diagonal(inputs, dim1=-2, dim2=-1)   # N x D x m
#         print(f'diag_part shape : {diag_part.shape}')
        sum_diag_part = torch.sum(diag_part, dim=2, keepdim=True)  # N x D x 1
#         print(f'sum_diag_part shape : {sum_diag_part.shape}')
        sum_of_rows = torch.sum(inputs, dim=3)  # N x D x m
#         print(f'sum_of_rows shape : {sum_of_rows.shape}')
        sum_of_cols = torch.sum(inputs, dim=2)  # N x D x m
#         print(f'sum_of_cols shape : {sum_of_cols.shape}')
        sum_all = torch.sum(sum_of_rows, dim=2)  # N x D
#         print(f'sum_all shape : {sum_all.shape}')

        # op1 - (1234) - extract diag
        op1 = torch.diag_embed(diag_part)  # N x D x m x m

        # op2 - (1234) + (12)(34) - place sum of diag on diag
        op2 = torch.diag_embed(sum_diag_part.repeat(1, 1, dim))  # N x D x m x m

        # op3 - (1234) + (123)(4) - place sum of row i on diag ii
        op3 = torch.diag_embed(sum_of_rows)  # N x D x m x m

        # op4 - (1234) + (124)(3) - place sum of col i on diag ii
        op4 = torch.diag_embed(sum_of_cols)  # N x D x m x m

        # op5 - (1234) + (124)(3) + (123)(4) + (12)(34) + (12)(3)(4) - place sum of all entries on diag
        op5 = torch.diag_embed(torch.unsqueeze(sum_all, dim=2).repeat(1, 1, dim))  # N x D x m x m

        # op6 - (14)(23) + (13)(24) + (24)(1)(3) + (124)(3) + (1234) - place sum of col i on row i
        op6 = torch.unsqueeze(sum_of_cols, dim=3).repeat(1, 1, 1, dim)  # N x D x m x m

        # op7 - (14)(23) + (23)(1)(4) + (234)(1) + (123)(4) + (1234) - place sum of row i on row i
        op7 = torch.unsqueeze(sum_of_rows, dim=3).repeat(1, 1, 1, dim)  # N x D x m x m

        # op8 - (14)(2)(3) + (134)(2) + (14)(23) + (124)(3) + (1234) - place sum of col i on col i
        op8 = torch.unsqueeze(sum_of_cols, dim=2).repeat(1, 1, dim, 1)  # N x D x m x m

        # op9 - (13)(24) + (13)(2)(4) + (134)(2) + (123)(4) + (1234) - place sum of row i on col i
        op9 = torch.unsqueeze(sum_of_rows, dim=2).repeat(1, 1, dim, 1)  # N x D x m x m

        # op10 - (1234) + (14)(23) - identity
        op10 = inputs  # N x D x m x m

        # op11 - (1234) + (13)(24) - transpose
        op11 = inputs.permute(0, 1, 3, 2)  # N x D x m x m

        # op12 - (1234) + (234)(1) - place ii element in row i
        op12 = torch.unsqueeze(diag_part, dim=3).repeat(1, 1, 1, dim)  # N x D x m x m

        # op13 - (1234) + (134)(2) - place ii element in col i
        op13 = torch.unsqueeze(diag_part, dim=2).repeat(1, 1, dim, 1)  # N x D x m x m

        # op14 - (34)(1)(2) + (234)(1) + (134)(2) + (1234) + (12)(34) - place sum of diag in all entries
        op14 = torch.unsqueeze(sum_diag_part, dim=3).repeat(1, 1, dim, dim)   # N x D x m x m

        # op15 - sum of all ops - place sum of all entries in all entries
        op15 = torch.unsqueeze(torch.unsqueeze(sum_all, dim=2), dim=3).repeat(1, 1, dim, dim)  # N x D x m x m

        if normalization is not None:
            float_dim = dim.type(torch.FloatTensor)
            if normalization is 'inf':
                op2 = torch.div(op2, float_dim)
                op3 = torch.div(op3, float_dim)
                op4 = torch.div(op4, float_dim)
                op5 = torch.div(op5, float_dim**2)
                op6 = torch.div(op6, float_dim)
                op7 = torch.div(op7, float_dim)
                op8 = torch.div(op8, float_dim)
                op9 = torch.div(op9, float_dim)
                op14 = torch.div(op14, float_dim)
                op15 = torch.div(op15, float_dim**2)

        return [op1, op2, op3, op4, op5, op6, op7, op8, op9, op10, op11, op12, op13, op14, op15]

    def forward(self, inputs, normalization='inf'):
        m = torch.tensor(inputs.shape[3], dtype=torch.int32, device=self.device)  # extract dimension

#         print(f'inputs device : {inputs.device}')
        ops_out = self.ops_2_to_2(inputs=inputs, dim=m, normalization=normalization)
#         for idx, op in enumerate(ops_out):
#             print(f'ops_out{idx} : {op.shape}')
        ops_out = torch.stack(ops_out, dim=2)

#         print(f'self.coeffs device : {self.coeffs.device}')
#         print(f'ops_out device : {ops_out.device}')
        output = torch.einsum('dsb,ndbij->nsij', self.coeffs.double(), ops_out)  # N x S x m x m

        # bias
#         print(f'diag_bias shape : {self.diag_bias.shape}')
#         print(f'eye shape : {torch.eye(torch.tensor(inputs.shape[3], dtype=torch.int32, device=self.device), device=self.device).shape}')
#         mat_diag_bias = torch.mul(torch.unsqueeze(torch.unsqueeze(torch.eye(torch.tensor(inputs.shape[3], dtype=torch.int32, device=self.device), device=self.device), 0), 0), self.diag_bias)
        mat_diag_bias = self.diag_bias.expand(-1,-1,inputs.shape[3],inputs.shape[3])
        mat_diag_bias = torch.mul(mat_diag_bias, torch.eye(inputs.shape[3], device=self.device))
        output = output + self.all_bias + mat_diag_bias
#         print(f'mat_diag_bias shape : {mat_diag_bias.shape}')

        return output

# def equi_2_to_2(name, input_depth, output_depth, inputs, normalization='inf', normalization_val=1.0):
#     '''
#     :param name: name of layer
#     :param input_depth: D
#     :param output_depth: S
#     :param inputs: N x D x m x m tensor
#     :return: output: N x S x m x m tensor
#     '''
#     basis_dimension = 15

#     # initialization values for variables
#     coeffs_values = torch.matmul(torch.randn(size=(input_depth, output_depth, basis_dimension), dtype=torch.float32), torch.sqrt(2. / (input_depth + output_depth).type(torch.FloatTensor)))
# #     coeffs_values = tf.multiply(tf.random_normal([input_depth, output_depth, basis_dimension], dtype=tf.float32), tf.sqrt(2. / tf.to_float(input_depth + output_depth)))
#     #coeffs_values = tf.random_normal([input_depth, output_depth, basis_dimension], dtype=tf.float32)
#     # define variables
#     coeffs = torch.autograd.Variable(coeffs_values, requires_grad=True)
# #     coeffs = tf.get_variable('coeffs', initializer=coeffs_values)

#     m = inputs.shape[3].type(torch.IntTensor)  # extract dimension
# #     m = tf.to_int32(tf.shape(inputs)[3])  # extract dimension

#     ops_out = ops_2_to_2(inputs, m, normalization=normalization)
#     ops_out = torch.stack(ops_out, dim=2)
# #     ops_out = tf.stack(ops_out, axis=2)

#     output = torch.einsum('dsb,ndbij->nsij', coeffs, ops_out)  # N x S x m x m
# #     output = tf.einsum('dsb,ndbij->nsij', coeffs, ops_out)  # N x S x m x m

#     # bias
#     diag_bias = torch.autograd.Variable(torch.zeros((1, output_depth, 1, 1), dtype=torch.float32), requires_grad=True)
# #     diag_bias = tf.get_variable('diag_bias', initializer=tf.zeros([1, output_depth, 1, 1], dtype=tf.float32))
#     all_bias = torch.autograd.Variable(torch.zeros((1, output_depth, 1, 1), dtype=torch.float32), requires_grad=True)
# #     all_bias = tf.get_variable('all_bias', initializer=tf.zeros([1, output_depth, 1, 1], dtype=tf.float32))
#     mat_diag_bias = torch.matmul(torch.unsqueeze(torch.unsqueeze(torch.eye(inputs.shape[3].type(torch.IntTensor)), 0), 0), diag_bias)
# #     mat_diag_bias = tf.multiply(tf.expand_dims(tf.expand_dims(tf.eye(tf.to_int32(tf.shape(inputs)[3])), 0), 0), diag_bias)
#     output = output + all_bias + mat_diag_bias

#     return output


def equi_2_to_1(name, input_depth, output_depth, inputs, normalization='inf', normalization_val=1.0):
    '''
    :param name: name of layer
    :param input_depth: D
    :param output_depth: S
    :param inputs: N x D x m x m tensor
    :return: output: N x S x m tensor
    '''
    basis_dimension = 5
#     with tf.variable_scope(name, reuse=tf.AUTO_REUSE) as scope:

    # initialization values for variables
    coeffs_values = torch.matmul(torch.randn(size=(input_depth, output_depth, basis_dimension), dtype=torch.float32), torch.sqrt(2. / (input_depth + output_depth).type(torch.FloatTensor)))
#     coeffs_values = tf.multiply(tf.random_normal([input_depth, output_depth, basis_dimension], dtype=tf.float32), tf.sqrt(2. / tf.to_float(input_depth + output_depth)))
    #coeffs_values = tf.random_normal([input_depth, output_depth, basis_dimension], dtype=tf.float32)
    # define variables
    coeffs = torch.autograd.Variable(coeffs_values, requires_grad=True)
#     coeffs = tf.get_variable('coeffs', initializer=coeffs_values)

    m = inputs.shape[3].type(torch.IntTensor)  # extract dimension
#     m = tf.to_int32(tf.shape(inputs)[3])  # extract dimension

    ops_out = ops_2_to_1(inputs, m, normalization=normalization)
    ops_out = torch.stack(ops_out, dim=2)
#     ops_out = tf.stack(ops_out, axis=2)  # N x D x B x m

    output = torch.einsum('dsb,ndbi->nsi', coeffs, ops_out)  # N x S x m x m
#     output = tf.einsum('dsb,ndbi->nsi', coeffs, ops_out)  # N x S x m

    # bias
    bias = torch.autograd.Variable(torch.zeros((1, output_depth, 1), dtype=torch.float32), requires_grad=True)
#     bias = tf.get_variable('bias', initializer=tf.zeros([1, output_depth, 1], dtype=tf.float32))
    output = output + bias

    return output


def equi_1_to_2(name, input_depth, output_depth, inputs, normalization='inf', normalization_val=1.0):
    '''
    :param name: name of layer
    :param input_depth: D
    :param output_depth: S
    :param inputs: N x D x m tensor
    :return: output: N x S x m x m tensor
    '''
    basis_dimension = 5
#     with tf.variable_scope(name, reuse=tf.AUTO_REUSE) as scope:

    # initialization values for variables
    coeffs_values = torch.matmul(torch.randn(size=(input_depth, output_depth, basis_dimension), dtype=torch.float32), torch.sqrt(2. / (input_depth + output_depth).type(torch.FloatTensor)))
#     coeffs_values = tf.multiply(tf.random_normal([input_depth, output_depth, basis_dimension], dtype=tf.float32), tf.sqrt(2. / tf.to_float(input_depth + output_depth)))
    #coeffs_values = tf.random_normal([input_depth, output_depth, basis_dimension], dtype=tf.float32)
    # define variables
    coeffs = torch.autograd.Variable(coeffs_values, requires_grad=True)
#     coeffs = tf.get_variable('coeffs', initializer=coeffs_values)

    m = inputs.shape[3].type(torch.IntTensor)  # extract dimension
#     m = tf.to_int32(tf.shape(inputs)[2])  # extract dimension

    ops_out = ops_1_to_2(inputs, m, normalization=normalization)
    ops_out = torch.stack(ops_out, dim=2)
#     ops_out = tf.stack(ops_out, axis=2)  # N x D x B x m x m

    output = torch.einsum('dsb,ndbij->nsij', coeffs, ops_out)  # N x S x m x m
#     output = tf.einsum('dsb,ndbij->nsij', coeffs, ops_out)  # N x S x m x m

    # bias
    bias = torch.autograd.Variable(torch.zeros((1, output_depth, 1, 1), dtype=torch.float32), requires_grad=True)
#     bias = tf.get_variable('bias', initializer=tf.zeros([1, output_depth, 1, 1], dtype=tf.float32))
    output = output + bias

    return output


def equi_1_to_1(name, input_depth, output_depth, inputs, normalization='inf', normalization_val=1.0):
    '''
    :param name: name of layer
    :param input_depth: D
    :param output_depth: S
    :param inputs: N x D x m tensor
    :return: output: N x S x m tensor
    '''
    basis_dimension = 2
#     with tf.variable_scope(name, reuse=tf.AUTO_REUSE) as scope:

    # initialization values for variables
    coeffs_values = torch.matmul(torch.randn(size=(input_depth, output_depth, basis_dimension), dtype=torch.float32), torch.sqrt(2. / (input_depth + output_depth).type(torch.FloatTensor)))
#     coeffs_values = tf.multiply(tf.random_normal([input_depth, output_depth, basis_dimension], dtype=tf.float32), tf.sqrt(2. / tf.to_float(input_depth + output_depth)))
    #coeffs_values = tf.random_normal([input_depth, output_depth, basis_dimension], dtype=tf.float32)
    # define variables
    coeffs = torch.autograd.Variable(coeffs_values, requires_grad=True)
#     coeffs = tf.get_variable('coeffs', initializer=coeffs_values)

    m = inputs.shape[3].type(torch.IntTensor)  # extract dimension
#     m = tf.to_int32(tf.shape(inputs)[2])  # extract dimension

    ops_out = ops_1_to_1(inputs, m, normalization=normalization)
    ops_out = torch.stack(ops_out, dim=2)
#     ops_out = tf.stack(ops_out, axis=2)  # N x D x B x m

    output = torch.einsum('dsb,ndbi->nsi', coeffs, ops_out)  # N x S x m x m
#     output = tf.einsum('dsb,ndbi->nsi', coeffs, ops_out)  # N x S x m

    # bias
    bias = torch.autograd.Variable(torch.zeros((1, output_depth, 1), dtype=torch.float32), requires_grad=True)
#     bias = tf.get_variable('bias', initializer=tf.zeros([1, output_depth, 1], dtype=tf.float32))
    output = output + bias

    return output


def equi_basic(name, input_depth, output_depth, inputs):
    '''
    :param name: name of layer
    :param input_depth: D
    :param output_depth: S
    :param inputs: N x D x m x m tensor
    :return: output: N x S x m x m tensor
    '''
    basis_dimension = 4
#     with tf.variable_scope(name, reuse=tf.AUTO_REUSE) as scope:

    # initialization values for variables
    coeffs_values = torch.matmul(torch.randn(size=(input_depth, output_depth, basis_dimension), dtype=torch.float32), torch.sqrt(2. / (input_depth + output_depth).type(torch.FloatTensor)))
#     coeffs_values = tf.multiply(tf.random_normal([input_depth, output_depth, basis_dimension], dtype=tf.float32), tf.sqrt(2. / tf.to_float(input_depth + output_depth)))
    #coeffs_values = tf.random_normal([input_depth, output_depth, basis_dimension], dtype=tf.float32)
    # define variables
    coeffs = torch.autograd.Variable(coeffs_values, requires_grad=True)
#     coeffs = tf.get_variable('coeffs', initializer=coeffs_values)

    m = inputs.shape[3].type(torch.IntTensor)  # extract dimension
#     m = tf.to_int32(tf.shape(inputs)[3])  # extract dimension
    float_dim = m.type(torch.FloatTensor)
#     float_dim = tf.to_float(m)


    # apply ops
    ops_out = []
    # w1 - identity
    ops_out.append(inputs)
    # w2 - sum cols
    sum_of_cols = torch.divide(torch.sum(inputs, dim=2), float_dim)  # N x D x m
#     sum_of_cols = tf.divide(tf.reduce_sum(inputs, axis=2), float_dim)  # N x D x m
    ops_out.append(torch.unsqueeze(sum_of_cols, dim=2).repeat(1, 1, m, 1))  # N x D x m x m
#     ops_out.append(tf.tile(tf.expand_dims(sum_of_cols, axis=2), [1, 1, m, 1]))  # N x D x m x m
    # w3 - sum rows
    sum_of_rows = torch.divide(torch.sum(inputs, dim=3), float_dim)  # N x D x m
#     sum_of_rows = tf.divide(tf.reduce_sum(inputs, axis=3), float_dim)  # N x D x m
    ops_out.append(torch.unsqueeze(sum_of_rows, dim=3).repeat(1, 1, 1, m))  # N x D x m x m
#     ops_out.append(tf.tile(tf.expand_dims(sum_of_rows, axis=3), [1, 1, 1, m]))  # N x D x m x m
    # w4 - sum all
    sum_all = torch.divide(torch.sum(sum_of_rows, dim=2), torch.square(float_dim))  # N x D
#     sum_all = tf.divide(tf.reduce_sum(sum_of_rows, axis=2), tf.square(float_dim))  # N x D
    ops_out.append(torch.unsqueeze(torch.unsqueeze(sum_all, dim=2), dim=3).repeat(1, 1, m, m))  # N x D x m x m
#     ops_out.append(tf.tile(tf.expand_dims(tf.expand_dims(sum_all, axis=2), axis=3), [1, 1, m, m]))  # N x D x m x m

    ops_out = torch.stack(ops_out, dim=2)
#     ops_out = tf.stack(ops_out, axis=2)
    output = torch.einsum('dsb,ndbij->nsij', coeffs, ops_out)  # N x S x m x m
#     output = tf.einsum('dsb,ndbij->nsij', coeffs, ops_out)  # N x S x m x m

    # bias
    bias = torch.autograd.Variable(torch.zeros((1, output_depth, 1, 1), dtype=torch.float32), requires_grad=True)
#     bias = tf.get_variable('bias', initializer=tf.zeros([1, output_depth, 1, 1], dtype=tf.float32))
    output = output + bias

    return output


# def ops_2_to_2(inputs, dim, normalization='inf', normalization_val=1.0):  # N x D x m x m
#     diag_part = torch.diagonal(inputs)   # N x D x m
#     sum_diag_part = torch.sum(diag_part, dim=2, keepdim=True)  # N x D x 1
#     sum_of_rows = torch.sum(inputs, dim=3)  # N x D x m
#     sum_of_cols = torch.sum(inputs, dim=2)  # N x D x m
#     sum_all = torch.sum(sum_of_rows, dim=2)  # N x D

#     # op1 - (1234) - extract diag
#     op1 = torch.diagonal(diag_part)  # N x D x m x m

#     # op2 - (1234) + (12)(34) - place sum of diag on diag
#     op2 = torch.diagonal(sum_diag_part.repeat(1, 1, dim))  # N x D x m x m

#     # op3 - (1234) + (123)(4) - place sum of row i on diag ii
#     op3 = torch.diagonal(sum_of_rows)  # N x D x m x m

#     # op4 - (1234) + (124)(3) - place sum of col i on diag ii
#     op4 = torch.diagonal(sum_of_cols)  # N x D x m x m

#     # op5 - (1234) + (124)(3) + (123)(4) + (12)(34) + (12)(3)(4) - place sum of all entries on diag
#     op5 = torch.diagonal(torch.unsqueeze(sum_all, dim=2).repeat(1, 1, dim))  # N x D x m x m

#     # op6 - (14)(23) + (13)(24) + (24)(1)(3) + (124)(3) + (1234) - place sum of col i on row i
#     op6 = torch.unsqueeze(sum_of_cols, dim=3).repeat(1, 1, 1, dim)  # N x D x m x m

#     # op7 - (14)(23) + (23)(1)(4) + (234)(1) + (123)(4) + (1234) - place sum of row i on row i
#     op7 = torch.unsqueeze(sum_of_rows, dim=3).repeat(1, 1, 1, dim)  # N x D x m x m

#     # op8 - (14)(2)(3) + (134)(2) + (14)(23) + (124)(3) + (1234) - place sum of col i on col i
#     op8 = torch.unsqueeze(sum_of_cols, dim=2).repeat(1, 1, dim, 1)  # N x D x m x m

#     # op9 - (13)(24) + (13)(2)(4) + (134)(2) + (123)(4) + (1234) - place sum of row i on col i
#     op9 = torch.unsqueeze(sum_of_rows, dim=2).repeat(1, 1, dim, 1)  # N x D x m x m

#     # op10 - (1234) + (14)(23) - identity
#     op10 = inputs  # N x D x m x m

#     # op11 - (1234) + (13)(24) - transpose
#     op11 = inputs.permute(0, 1, 3, 2)  # N x D x m x m

#     # op12 - (1234) + (234)(1) - place ii element in row i
#     op12 = torch.unsqueeze(diag_part, dim=3).repeat(1, 1, 1, dim)  # N x D x m x m

#     # op13 - (1234) + (134)(2) - place ii element in col i
#     op13 = torch.unsqueeze(diag_part, dim=2).repeat(1, 1, dim, 1)  # N x D x m x m

#     # op14 - (34)(1)(2) + (234)(1) + (134)(2) + (1234) + (12)(34) - place sum of diag in all entries
#     op14 = torch.unsqueeze(sum_diag_part, dim=3).repeat(1, 1, dim, dim)   # N x D x m x m

#     # op15 - sum of all ops - place sum of all entries in all entries
#     op15 = torch.unsqueeze(torch.unsqueeze(sum_all, dim=2), dim=3).repeat(1, 1, dim, dim)  # N x D x m x m

#     if normalization is not None:
#         float_dim = dim.type(torch.FloatTensor)
#         if normalization is 'inf':
#             op2 = torch.div(op2, float_dim)
#             op3 = torch.div(op3, float_dim)
#             op4 = torch.div(op4, float_dim)
#             op5 = torch.div(op5, float_dim**2)
#             op6 = torch.div(op6, float_dim)
#             op7 = torch.div(op7, float_dim)
#             op8 = torch.div(op8, float_dim)
#             op9 = torch.div(op9, float_dim)
#             op14 = torch.div(op14, float_dim)
#             op15 = torch.div(op15, float_dim**2)

#     return [op1, op2, op3, op4, op5, op6, op7, op8, op9, op10, op11, op12, op13, op14, op15]


def ops_2_to_1(inputs, dim, normalization='inf', normalization_val=1.0):  # N x D x m x m
    diag_part = tf.matrix_diag_part(inputs)  # N x D x m
    sum_diag_part = tf.reduce_sum(diag_part, axis=2, keepdims=True)  # N x D x 1
    sum_of_rows = tf.reduce_sum(inputs, axis=3)  # N x D x m
    sum_of_cols = tf.reduce_sum(inputs, axis=2)  # N x D x m
    sum_all = tf.reduce_sum(inputs, axis=(2, 3))  # N x D

    # op1 - (123) - extract diag
    op1 = diag_part  # N x D x m

    # op2 - (123) + (12)(3) - tile sum of diag part
    op2 = tf.tile(sum_diag_part, [1, 1, dim])  # N x D x m

    # op3 - (123) + (13)(2) - place sum of row i in element i
    op3 = sum_of_rows  # N x D x m

    # op4 - (123) + (23)(1) - place sum of col i in element i
    op4 = sum_of_cols  # N x D x m

    # op5 - (1)(2)(3) + (123) + (12)(3) + (13)(2) + (23)(1) - tile sum of all entries
    op5 = tf.tile(tf.expand_dims(sum_all, axis=2), [1, 1, dim])  # N x D x m


    if normalization is not None:
        float_dim = tf.to_float(dim)
        if normalization is 'inf':
            op2 = tf.divide(op2, float_dim)
            op3 = tf.divide(op3, float_dim)
            op4 = tf.divide(op4, float_dim)
            op5 = tf.divide(op5, float_dim ** 2)

    return [op1, op2, op3, op4, op5]


def ops_1_to_2(inputs, dim, normalization='inf', normalization_val=1.0):  # N x D x m
    sum_all = tf.reduce_sum(inputs, axis=2, keepdims=True)  # N x D x 1

    # op1 - (123) - place on diag
    op1 = tf.matrix_diag(inputs)  # N x D x m x m

    # op2 - (123) + (12)(3) - tile sum on diag
    op2 = tf.matrix_diag(tf.tile(sum_all, [1, 1, dim]))  # N x D x m x m

    # op3 - (123) + (13)(2) - tile element i in row i
    op3 = tf.tile(tf.expand_dims(inputs, axis=2), [1, 1, dim, 1])  # N x D x m x m

    # op4 - (123) + (23)(1) - tile element i in col i
    op4 = tf.tile(tf.expand_dims(inputs, axis=3), [1, 1, 1, dim])  # N x D x m x m

    # op5 - (1)(2)(3) + (123) + (12)(3) + (13)(2) + (23)(1) - tile sum of all entries
    op5 = tf.tile(tf.expand_dims(sum_all, axis=3), [1, 1, dim, dim])  # N x D x m x m

    if normalization is not None:
        float_dim = tf.to_float(dim)
        if normalization is 'inf':
            op2 = tf.divide(op2, float_dim)
            op5 = tf.divide(op5, float_dim)

    return [op1, op2, op3, op4, op5]


def ops_1_to_1(inputs, dim, normalization='inf', normalization_val=1.0):  # N x D x m
    sum_all = tf.reduce_sum(inputs, axis=2, keepdims=True)  # N x D x 1

    # op1 - (12) - identity
    op1 = inputs  # N x D x m

    # op2 - (1)(2) - tile sum of all
    op2 = tf.tile(sum_all, [1, 1, dim])  # N x D x m

    if normalization is not None:
        float_dim = tf.to_float(dim)
        if normalization is 'inf':
            op2 = tf.divide(op2, float_dim)

    return [op1, op2]

