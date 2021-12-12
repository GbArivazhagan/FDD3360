#include <iostream>
#include <math.h>
#include <time.h>
#include <wchar.h>

#define NUM_PARTICLES 100000
#define NUM_ITERATIONS 25000
#define BLOCK_SIZE 256
#define BATCH_SIZE 100000
#define NUM_STREAMS 1

typedef struct Particle {
    float3 position;
    float3 velocity;
} Particle;

void init_position(Particle *cpu_particles, int i)
{
      cpu_particles[i].position.x = i*0.001;
      cpu_particles[i].position.y = (i*i - 2*i)*0.001;
      cpu_particles[i].position.z = 0.00005*(i*i + 3*i);
}

void cpu_position_update(Particle *cpu_particles, float dt)
{
    for(int i = 0; i < NUM_PARTICLES; i++)
    {
        cpu_particles[i].position.x = cpu_particles[i].position.x + dt*cpu_particles[i].velocity.x;
        cpu_particles[i].position.y = cpu_particles[i].position.y + dt*cpu_particles[i].velocity.y;
        cpu_particles[i].position.z = cpu_particles[i].position.z + dt*cpu_particles[i].velocity.z;
        
    }
}

void cpu_velocity_update(Particle *cpu_particles, float t)
{
    for(int i = 0; i < NUM_PARTICLES; i++)
    {
        cpu_particles[i].velocity.x = 0.00001*t - 0.0005*cpu_particles[i].position.x;
        cpu_particles[i].velocity.y = 0.00001*t - 0.0002*cpu_particles[i].position.x*cpu_particles[i].position.y;
        cpu_particles[i].velocity.z = 0.00001*t - 0.0001*cpu_particles[i].position.x*cpu_particles[i].position.z;
    }
}

void cpu_particle(Particle *cpu_particles, float dt)
{
    for(int i = 0; i < NUM_ITERATIONS; i++)
    {
        cpu_velocity_update(cpu_particles, (i+1)*dt);
        cpu_position_update(cpu_particles, dt);
    }
}

__device__
void gpu_velocity_update(Particle *gpu_particles, float t, int index)
{
        gpu_particles[index].velocity.x = 0.00001*t - 0.0005*gpu_particles[index].position.x;
        gpu_particles[index].velocity.y = 0.00001*t - 0.0002*gpu_particles[index].position.x*gpu_particles[index].position.y;
        gpu_particles[index].velocity.z = 0.00001*t - 0.0001*gpu_particles[index].position.x*gpu_particles[index].position.z;
}

__device__
void gpu_position_update(Particle *gpu_particles, float dt, int index)
{
        gpu_particles[index].position.x = gpu_particles[index].position.x + dt*gpu_particles[index].velocity.x;
        gpu_particles[index].position.y = gpu_particles[index].position.y + dt*gpu_particles[index].velocity.y;
        gpu_particles[index].position.z = gpu_particles[index].position.z + dt*gpu_particles[index].velocity.z;
}

__global__
void gpu_particle(Particle *gpu_particles, float dt)
{
    int j = blockIdx.x*blockDim.x + threadIdx.x;
    if (j < NUM_PARTICLES)
    {
        for(int i = 0; i < NUM_ITERATIONS; i++)
        {
            gpu_velocity_update(gpu_particles, (i+1)*dt, j);
            gpu_position_update(gpu_particles, dt, j);
        }
    }
}

__global__
void gpu_particle_serial(Particle *gpu_particles, int t, float dt)
{
    int j = blockIdx.x*blockDim.x + threadIdx.x;
    if (j < NUM_PARTICLES)
    {
//        for(int i = 0; i < NUM_ITERATIONS; i++)
//        {
            gpu_velocity_update(gpu_particles, t, j);
            gpu_position_update(gpu_particles, dt, j);
//        }
    }
}

__global__
void gpu_particle_sync(Particle *gpu_particles, int offset, int t, float dt)
{
    int j = blockIdx.x*blockDim.x + threadIdx.x + offset;
    if (j < NUM_PARTICLES)
    {
//        for(int i = 0; i < NUM_ITERATIONS; i++)
//        {
            gpu_velocity_update(gpu_particles, t, j);
            gpu_position_update(gpu_particles, dt, j);
//        }
    }
}


void init(Particle *cpu_particles)
{
    for(int i = 0; i < NUM_PARTICLES; i++)
    {
        init_position(cpu_particles, i);
    }
}

int main()
{
    // Initiate data structures
    Particle *cpu_particles;
    Particle *cpu_particles_init;
    Particle *gpu_particles;
    Particle *gpu_particles_sync;
    Particle *gpu_particles_sync_2;
    Particle *gpu_particles_serial;
    Particle *res_particles;
    
    clock_t start, stop;
    double elapsed;
    
    float dt = 1.0;
    
    cudaMalloc(&gpu_particles, NUM_PARTICLES*sizeof(Particle));
    cudaMalloc(&gpu_particles_sync, NUM_PARTICLES*sizeof(Particle));
    cudaMalloc(&gpu_particles_sync_2, NUM_PARTICLES*sizeof(Particle));
    cudaMalloc(&gpu_particles_serial, NUM_PARTICLES*sizeof(Particle));
    cudaHostAlloc(&res_particles, NUM_PARTICLES*sizeof(Particle),cudaHostAllocDefault);
    cudaHostAlloc(&cpu_particles_init, NUM_PARTICLES*sizeof(Particle),cudaHostAllocDefault);
    cudaHostAlloc(&cpu_particles, NUM_PARTICLES*sizeof(Particle),cudaHostAllocDefault);

    init(cpu_particles_init);
    
    cpu_particles = cpu_particles_init;
    cudaMemcpy(&gpu_particles,&cpu_particles,NUM_PARTICLES*sizeof(Particle),cudaMemcpyHostToDevice);
/*
    clock_t start = clock();
    cpu_particle(cpu_particles, dt);
    clock_t stop = clock();
    double elapsed = (double)(stop - start) * 1000.0 / CLOCKS_PER_SEC;
    printf("CPU time elapsed in ms: %f\n", elapsed);
    
    start = clock();
    gpu_particle<<<(NUM_PARTICLES + BLOCK_SIZE - 1) / BLOCK_SIZE,BLOCK_SIZE>>>(gpu_particles, dt);
    cudaDeviceSynchronize();
    stop = clock();
    elapsed = (double)(stop - start) * 1000.0 / CLOCKS_PER_SEC;
    printf("GPU time elapsed in ms: %f\n", elapsed);
    
    start = clock();
    for (int i = 0; i < NUM_ITERATIONS; i++)
    {
        gpu_particle_serial<<<(NUM_PARTICLES + BLOCK_SIZE - 1) / BLOCK_SIZE,BLOCK_SIZE>>>(gpu_particles_serial, i+1, dt);
    }
    cudaDeviceSynchronize();
    stop = clock();
    elapsed = (double)(stop - start) * 1000.0 / CLOCKS_PER_SEC;
    printf("GPU Serial time elapsed in ms: %f\n", elapsed);
*/
    cudaMemcpy(&res_particles,&gpu_particles,NUM_PARTICLES*sizeof(Particle),cudaMemcpyDeviceToHost);
    
    cudaStream_t streams[NUM_STREAMS];
    const int stream_particle_count = NUM_PARTICLES/NUM_STREAMS;
    const int stream_loops = (stream_particle_count + BATCH_SIZE - 1)/BATCH_SIZE;
//    printf("stream_loops %d \n", stream_loops);
//    printf("BATCH_SIZE %d\n", BATCH_SIZE);
    const int batch_bytes = BATCH_SIZE * sizeof(Particle);
/*
    for (int i = 0; i < NUM_STREAMS; i++)
    {
        cudaStreamCreate(&streams[i]);
    }
    
    start = clock();
    for(int t = 0; t < NUM_ITERATIONS; t++)
    {
        for (int i = 0; i < NUM_STREAMS; i++)
        {
            cudaStreamSynchronize(streams[i]);
            for (int k = 0; k < stream_loops; k++)
            {
//                printf("Stream: %d, Start of stream loop %d\n", i, k);
                int offset = i * stream_particle_count + k * BATCH_SIZE;
                cudaMemcpyAsync(&gpu_particles_sync[offset], &cpu_particles_init[offset], batch_bytes, cudaMemcpyHostToDevice, streams[i]);
                gpu_particle_sync<<<(BATCH_SIZE + BLOCK_SIZE - 1)/BLOCK_SIZE, BLOCK_SIZE, 0, streams[i]>>>(gpu_particles_sync, offset, t, dt);
                cudaMemcpyAsync(&res_particles[offset], &gpu_particles_sync[offset], batch_bytes, cudaMemcpyDeviceToHost, streams[i]);
            }
        }
    }
    stop = clock();
    elapsed = (double)(stop - start) * 1000.0 / CLOCKS_PER_SEC;
    printf("GPU sync time elapsed in ms: %f\n", elapsed);
    
    for (int i = 0; i < NUM_STREAMS; i++)
    {
        cudaStreamDestroy(streams[i]);
    }
*/
    for (int i = 0; i < NUM_STREAMS; i++)
    {
        cudaStreamCreate(&streams[i]);
    }
    
    start = clock();
    for(int t = 0; t < NUM_ITERATIONS; t++)
    {
        for (int i = 0; i < NUM_STREAMS; i++)
        {
//            cudaStreamSynchronize(streams[i]);
            for (int k = 0; k < stream_loops; k++)
            {
//                printf("Stream: %d, Start of stream loop %d\n", i, k);
                int offset = i * stream_particle_count + k * BATCH_SIZE;
                cudaMemcpyAsync(&gpu_particles_sync[offset], &cpu_particles_init[offset], batch_bytes, cudaMemcpyHostToDevice, streams[i]);
            }
        }
        for (int i = 0; i < NUM_STREAMS; i++)
        {
//            cudaStreamSynchronize(streams[i]);
            for (int k = 0; k < stream_loops; k++)
            {
//                printf("Stream: %d, Start of stream loop %d\n", i, k);
                int offset = i * stream_particle_count + k * BATCH_SIZE;
                gpu_particle_sync<<<(BATCH_SIZE + BLOCK_SIZE - 1)/BLOCK_SIZE, BLOCK_SIZE, 0, streams[i]>>>(gpu_particles_sync, offset, t, dt);
            }
        }
        for (int i = 0; i < NUM_STREAMS; i++)
        {
//            cudaStreamSynchronize(streams[i]);
            for (int k = 0; k < stream_loops; k++)
            {
//                printf("Stream: %d, Start of stream loop %d\n", i, k);
                int offset = i * stream_particle_count + k * BATCH_SIZE;
                cudaMemcpyAsync(&res_particles[offset], &gpu_particles_sync[offset], batch_bytes, cudaMemcpyDeviceToHost, streams[i]);
            }
        }
    }
    stop = clock();
    elapsed = (double)(stop - start) * 1000.0 / CLOCKS_PER_SEC;
    printf("GPU Sync 2 time elapsed in ms: %f\n", elapsed);
    
    for (int i = 0; i < NUM_STREAMS; i++)
    {
        cudaStreamDestroy(streams[i]);
    }
    
    cudaMemcpy(&res_particles,&gpu_particles_sync_2,NUM_PARTICLES*sizeof(Particle),cudaMemcpyDeviceToHost);
        
    cudaFree(cpu_particles);
    cudaFree(cpu_particles_init);
    cudaFree(gpu_particles);
    cudaFree(gpu_particles_sync);
    cudaFree(gpu_particles_sync_2);
    cudaFree(gpu_particles_serial);
    cudaFree(res_particles);
}
