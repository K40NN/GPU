#include "matrix.h"
#include <stdlib.h>
#include <string.h>

#define MIN(a,b) (((a)<(b))?(a):(b))
#define MAX(a,b) (((a)>(b))?(a):(b))

matrix_t * alloc_matrix(unsigned rows, unsigned columns)
{
    matrix_t * res = (matrix_t*) malloc( sizeof(matrix_t) );
    res->m = (double *) calloc(columns * rows, sizeof(double));
    res->columns = columns;
    res->rows = rows;
    res->size = rows * columns * sizeof(double);
    return res;
}

void destroy_matrix(matrix_t *m)
{
    //printf("free %p %p\n", m, m->m);
    free(m->m);
    free(m);
}

void print_matrix(matrix_t *m, bool is_short){
    unsigned lim_rows = 0;
    unsigned lim_col = 0;

    if (is_short)
    {
        lim_rows = MIN(m->rows, 4);
        lim_col = MIN(m->columns, 10);
    }
    else
    {
        lim_rows = m->rows;
        lim_col = m->columns;
    }

    for (int row = 0; row < lim_rows; row ++)
    {
        for (int col = 0; col < lim_col; col ++)
        {
            printf("%.2lf ", m->m[col + row * m->columns]);
        }
        if (is_short && lim_col != m->columns) printf("...");
        printf("\n");
    }
    if (is_short && lim_rows != m->rows) printf("...\n");
}

void hadamard_product(matrix_t *m1, matrix_t *m2, matrix_t *res)
{
    assert ( (m1->columns == m2->columns)   &&
             (m1->columns == res->columns)  &&
             (m1->rows == m2->rows)         &&
             (m1->rows == res->rows));

    for (int idx = 0; idx < m1->rows * m1->columns; idx ++)
    {
            res->m[idx] = m1->m[idx] * m2->m[idx];
    }
}

void matrix_sum(matrix_t *m1, matrix_t *m2, matrix_t *res)
{
    assert ( (m1->columns == m2->columns)  &&
             (m1->columns == res->columns) &&
             (m1->rows == m2->rows)        &&
             (m1->rows == res->rows));

    for (int idx = 0; idx < m1->rows * m1->columns; idx ++)
    { 
        res->m[idx] = m1->m[idx] + m2->m[idx];
    }
}

void matrix_minus(matrix_t *m1, matrix_t *m2, matrix_t *res)
{
    assert ( (m1->columns == m2->columns)  &&
             (m1->columns == res->columns) &&
             (m1->rows == m2->rows)        &&
             (m1->rows == res->rows));
             
    for (int idx = 0; idx < m1->rows * m1->columns; idx ++)
    {
        res->m[idx] = m1->m[idx] - m2->m[idx];
    }
}

__global__ void computeMatrixMulGPU(double *A, double *B, double *C, int numARows, int numAColumns, int numBRows, int numBColumns) 
{
    int indexx = blockDim.x * blockIdx.x + threadIdx.x;
    int indexy = blockDim.y * blockIdx.y + threadIdx.y;

    if (indexx < numBColumns && indexy < numARows) {
        double sum = 0.0;
        for (int k = 0; k < numAColumns; k++) {
            sum += A[indexy * numAColumns + k] * B[k * numBColumns + indexx];
        }
        C[indexy * numBColumns + indexx] = sum;
    }
}

void matrix_dot(matrix_t *m1, matrix_t *m2, matrix_t *res)
{
    assert((m1->columns == m2->rows) &&
           (m1->rows == res->rows) &&
           (m2->columns == res->columns));

    double *deviceA, *deviceB, *deviceC;

    // Allocate memory on the GPU
    cudaMalloc((void **)&deviceA, m1->rows * m1->columns * sizeof(double));
    cudaMalloc((void **)&deviceB, m2->rows * m2->columns * sizeof(double));
    cudaMalloc((void **)&deviceC, res->rows * res->columns * sizeof(double));

    // Copy data from host to device
    cudaMemcpy(deviceA, m1->m, m1->rows * m1->columns * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(deviceB, m2->m, m2->rows * m2->columns * sizeof(double), cudaMemcpyHostToDevice);

    // Set up grid and block dimensions
    dim3 blockDim(16, 16);
    dim3 gridDim((res->columns + blockDim.x - 1) / blockDim.x, (res->rows + blockDim.y - 1) / blockDim.y);

    // Launch the kernel
    computeMatrixMulGPU<<<gridDim, blockDim>>>(deviceA, deviceB, deviceC, m1->rows, m1->columns, m2->rows, m2->columns);

    // Copy the result back to the host
    cudaMemcpy(res->m, deviceC, res->rows * res->columns * sizeof(double), cudaMemcpyDeviceToHost);

    // Free GPU memory
    cudaFree(deviceA);
    cudaFree(deviceB);
    cudaFree(deviceC);
}

void matrix_function(matrix_t *m1, double (*f)(double), matrix_t *res)
{
    assert ( (m1->columns == res->columns) &&             
             (m1->rows == res->rows));

    for (int idx = 0; idx < m1->rows * m1->columns; idx ++)
    {
        res->m[idx] = f(m1->m[idx]);
    }
}

void matrix_transpose(matrix_t *m1, matrix_t *res)
{
    assert ( (m1->columns == res->rows) &&             
             (m1->rows == res->columns));
    
    for (int row = 0; row < m1->rows; row++)
    {
        for (int col = 0; col < m1->columns; col ++)
        {
            res->m[row + col * m1->rows] = m1->m[col + row * m1->columns];
        }
    }
}

void matrix_scalar(matrix_t *m1, double s, matrix_t *res)
{
    assert ( (m1->rows == res->rows) &&             
             (m1->columns == res->columns));

    for (int idx = 0; idx < m1->columns*m1->rows; idx ++)
    {
        res->m[idx] = m1->m[idx] * s;
    }
}

void matrix_memcpy(matrix_t *dest, const matrix_t *src)
{
    assert ( (dest->rows == src->rows)      &&             
             (dest->columns == src->columns));

    memcpy(dest->m, src->m, src->columns * src->rows * sizeof(double));     
}