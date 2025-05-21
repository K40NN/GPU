#ifndef GPU_MEMORY_H
#define GPU_MEMORY_H

typedef struct {
    // Layer 1
    double *device_weights1;
    double *device_activations1;
    double *device_z1;
    double *device_biases1;
    double *device_one1;
    double *device_z2;

    // Layer 2
    double *device_weights2;
    double *device_activations2;
    double *device_z1_2;
    double *device_biases2;
    double *device_one2;
    double *device_z2_2;

} gpu_memory_t;

#endif
