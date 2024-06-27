#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include <time.h>

// PSO参数
#define MAX_ITERATIONS 1000
#define INERTIA_WEIGHT 0.729
#define COGNITIVE_WEIGHT 1.49445
#define SOCIAL_WEIGHT 1.49445
#define MIN_POSITION -10.0
#define MAX_POSITION 10.0
#define MIN_VELOCITY -2.0
#define MAX_VELOCITY 2.0

// 粒子结构
typedef struct {
    double *position;
    double *velocity;
    double *best_position;
    double best_value;
} Particle;

// 随机数生成器
double random_double(double min, double max) {
    return min + (max - min) * ((double)rand() / RAND_MAX);
}

// 初始化粒子群
void initializeParticles(Particle *particles, int num_particles, int num_dimensions) {
    for (int i = 0; i < num_particles; ++i) {
        for (int j = 0; j < num_dimensions; ++j) {
            particles[i].position[j] = random_double(MIN_POSITION, MAX_POSITION);
            particles[i].velocity[j] = random_double(MIN_VELOCITY, MAX_VELOCITY);
            particles[i].best_position[j] = particles[i].position[j];
        }
        particles[i].best_value = DBL_MAX;
    }
}

// 计算适应度值
double objectiveFunction(double *position, int num_dimensions) {
    // 示例目标函数：Rosenbrock function
    double sum = 0.0;
    for (int i = 0; i < num_dimensions - 1; ++i) {
        double term1 = pow((position[i + 1] - position[i] * position[i]), 2.0);
        double term2 = pow((1 - position[i]), 2.0);
        sum += 100 * term1 + term2;
    }
    return sum;
}

// 更新粒子位置和速度
void updateParticle(Particle *particle, double *global_best_position, int num_dimensions) {
    for (int d = 0; d < num_dimensions; ++d) {
        double r1 = random_double(0.0, 1.0);
        double r2 = random_double(0.0, 1.0);

        particle->velocity[d] = INERTIA_WEIGHT * particle->velocity[d] +
                                COGNITIVE_WEIGHT * r1 * (particle->best_position[d] - particle->position[d]) +
                                SOCIAL_WEIGHT * r2 * (global_best_position[d] - particle->position[d]);

        particle->position[d] += particle->velocity[d];

        // 确保粒子位置在允许范围内
        if (particle->position[d] < MIN_POSITION) particle->position[d] = MIN_POSITION;
        if (particle->position[d] > MAX_POSITION) particle->position[d] = MAX_POSITION;
    }
}

// 粒子群优化算法
void pso(int rank, int size, int num_particles, int num_dimensions, int max_iterations, FILE *results) {
    int local_num_particles = num_particles / size;
    Particle *particles = (Particle *)malloc(local_num_particles * sizeof(Particle));
    for (int i = 0; i < local_num_particles; ++i) {
        particles[i].position = (double *)malloc(num_dimensions * sizeof(double));
        particles[i].velocity = (double *)malloc(num_dimensions * sizeof(double));
        particles[i].best_position = (double *)malloc(num_dimensions * sizeof(double));
    }
    initializeParticles(particles, local_num_particles, num_dimensions);

    double *global_best_position = (double *)malloc(num_dimensions * sizeof(double));
    double global_best_value = DBL_MAX;

    double comm_time = 0.0;

    for (int iter = 0; iter < max_iterations; ++iter) {
        for (int i = 0; i < local_num_particles; ++i) {
            double value = objectiveFunction(particles[i].position, num_dimensions);
            if (value < particles[i].best_value) {
                particles[i].best_value = value;
                for (int d = 0; d < num_dimensions; ++d) {
                    particles[i].best_position[d] = particles[i].position[d];
                }
            }
        }

        Particle local_best_particle = particles[0];
        for (int i = 1; i < local_num_particles; ++i) {
            if (particles[i].best_value < local_best_particle.best_value) {
                local_best_particle = particles[i];
            }
        }

        double local_best_value = local_best_particle.best_value;

        double start_comm = MPI_Wtime();
        MPI_Allreduce(&local_best_value, &global_best_value, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
        if (local_best_value == global_best_value) {
            for (int d = 0; d < num_dimensions; ++d) {
                global_best_position[d] = local_best_particle.best_position[d];
            }
        }
        double end_comm = MPI_Wtime();
        comm_time += (end_comm - start_comm) * 1000;

        for (int i = 0; i < local_num_particles; ++i) {
            updateParticle(&particles[i], global_best_position, num_dimensions);
        }
    }

    // 收集并输出最终的最佳结果
    if (rank == 0) {
        printf("Global Best Value: %f\n", global_best_value);
    }

    double total_time = comm_time + max_iterations * (num_particles / size) * num_dimensions;

    // 每个进程记录其运行时间和通信时间
    if (results != NULL) {
        fprintf(results, "%d,%d,%d,%.6f,%.6f,%.6f\n", size, num_particles, num_dimensions, total_time, comm_time, total_time - comm_time);
    }

    for (int i = 0; i < local_num_particles; ++i) {
        free(particles[i].position);
        free(particles[i].velocity);
        free(particles[i].best_position);
    }
    free(particles);
    free(global_best_position);
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int num_particles_list[] = {100, 200, 300,500,1000}; // 不同粒子数量规模
    int num_dimensions_list[] = {10, 20, 30,40,50};   // 不同维度规模

    // 打开CSV文件用于写入性能数据
    FILE *results = NULL;
    if (rank == 0) {
        results = fopen("pso_performance.csv","w");
		
        if (results != NULL) {
            fprintf(results, "Num_Processes,Num_Particles,Num_Dimensions,Total_Time(ms),Comm_Time(ms),Comp_Time(ms)\n");
        }
    }

    for (int i = 0; i < sizeof(num_particles_list) / sizeof(num_particles_list[0]); ++i) {
        for (int j = 0; j < sizeof(num_dimensions_list) / sizeof(num_dimensions_list[0]); ++j) {
            int num_particles = num_particles_list[i];
            int num_dimensions = num_dimensions_list[j];
            double start_time = MPI_Wtime();
            pso(rank, size, num_particles, num_dimensions, MAX_ITERATIONS, results);
            double end_time = MPI_Wtime();

            double elapsed_time = (end_time - start_time) * 1000;
            if (rank == 0) {
                printf("Total Time for Num_Particles=%d and Num_Dimensions=%d: %.6f ms\n", num_particles, num_dimensions, elapsed_time);
            }
        }
    }

    if (rank == 0 && results != NULL) {
        fclose(results);
        printf("Performance data written to pso_performance.csv\n");
    }

    MPI_Finalize();
    return 0;
}
