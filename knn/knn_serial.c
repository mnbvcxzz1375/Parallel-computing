#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <errno.h>

#define CHECK_ERROR(cond, msg) \
    do { if (cond) { perror(msg); exit(EXIT_FAILURE); } } while (0)

typedef struct {
    float *data;
    int label;
} Sample;

void readCSV(const char *filename, Sample **data, int *num_samples, int *num_features) {
    FILE *file = fopen(filename, "r");
    CHECK_ERROR(!file, "Error opening file");

    char line[1024];
    int capacity = 10;
    int n_samples = 0;
    *data = (Sample *)malloc(capacity * sizeof(Sample));
    CHECK_ERROR(!*data, "Error allocating memory");

    int feature_count = -1;  // Initialize feature count to -1

    while (fgets(line, 1024, file)) {
        char *token;
        int count = 0;
        Sample sample;
        sample.data = (float *)malloc(100 * sizeof(float)); // Assuming no more than 100 features
        CHECK_ERROR(!sample.data, "Error allocating memory");

        token = strtok(line, ",");
        while (token) {
            if (count == 99) {
                sample.data = (float *)realloc(sample.data, 200 * sizeof(float));
                CHECK_ERROR(!sample.data, "Error reallocating memory");
            }
            sample.data[count++] = atof(token);
            token = strtok(NULL, ",");
        }
        sample.label = (int)sample.data[count - 1];
        count--;
        sample.data = (float *)realloc(sample.data, count * sizeof(float));
        CHECK_ERROR(!sample.data, "Error reallocating memory");

        (*data)[n_samples++] = sample;
        if (n_samples == capacity) {
            capacity *= 2;
            *data = (Sample *)realloc(*data, capacity * sizeof(Sample));
            CHECK_ERROR(!*data, "Error reallocating memory");
        }

        if (feature_count == -1) {
            feature_count = count;  // Set feature count based on the first sample
        }
    }
    fclose(file);

    *num_samples = n_samples;
    *num_features = feature_count;  // Set num_features based on the count of the first sample
}

void splitDataset(Sample *data, int num_samples, Sample **train_data, int *num_train, Sample **test_data, int *num_test, float train_ratio) {
    int train_samples = (int)(num_samples * train_ratio);
    int test_samples = num_samples - train_samples;

    *train_data = (Sample *)malloc(train_samples * sizeof(Sample));
    *test_data = (Sample *)malloc(test_samples * sizeof(Sample));
    CHECK_ERROR(!*train_data || !*test_data, "Error allocating memory");

    int *indices = (int *)malloc(num_samples * sizeof(int));
    CHECK_ERROR(!indices, "Error allocating memory");

    for (int i = 0; i < num_samples; i++) indices[i] = i;
    srand((unsigned int)time(NULL));
    for (int i = 0; i < num_samples; i++) {
        int j = i + rand() / (RAND_MAX / (num_samples - i) + 1);
        int temp = indices[i];
        indices[i] = indices[j];
        indices[j] = temp;
    }

    for (int i = 0; i < train_samples; i++) {
        (*train_data)[i] = data[indices[i]];
    }
    for (int i = 0; i < test_samples; i++) {
        (*test_data)[i] = data[indices[train_samples + i]];
    }

    *num_train = train_samples;
    *num_test = test_samples;

    free(indices);
}

void computeDistances(Sample *train_data, int num_train, Sample *test_data, int num_test, int num_features, float **distances) {
    *distances = (float *)malloc(num_test * num_train * sizeof(float));
    CHECK_ERROR(!*distances, "Error allocating memory");

    for (int i = 0; i < num_test; i++) {
        for (int j = 0; j < num_train; j++) {
            float dist = 0.0;
            for (int k = 0; k < num_features; k++) {
                float diff = test_data[i].data[k] - train_data[j].data[k];
                dist += diff * diff;
            }
            (*distances)[i * num_train + j] = sqrtf(dist);
        }
    }
}

int comparePairs(const void *a, const void *b) {
    float diff = ((float *)a)[0] - ((float *)b)[0];
    return (diff > 0) - (diff < 0);
}

void classify(float *distances, Sample *train_data, int num_train, int num_test, int K, int *test_labels) {
    for (int i = 0; i < num_test; i++) {
        float dist_label_pairs[num_train][2];
        for (int j = 0; j < num_train; j++) {
            dist_label_pairs[j][0] = distances[i * num_train + j];
            dist_label_pairs[j][1] = train_data[j].label;
        }
        qsort(dist_label_pairs, num_train, sizeof(dist_label_pairs[0]), comparePairs);
        int k_nearest_labels[K];
        for (int k = 0; k < K; k++) {
            k_nearest_labels[k] = (int)dist_label_pairs[k][1];
        }
        qsort(k_nearest_labels, K, sizeof(int), comparePairs);

        int count = 1, max_count = 1, max_label = k_nearest_labels[0], current_label = k_nearest_labels[0];
        for (int k = 1; k < K; k++) {
            if (k_nearest_labels[k] == current_label) {
                count++;
            } else {
                count = 1;
                current_label = k_nearest_labels[k];
            }
            if (count > max_count) {
                max_count = count;
                max_label = current_label;
            }
        }
        test_labels[i] = max_label;
    }
}

float computeAccuracy(int *true_labels, int *predicted_labels, int num_samples) {
    int correct = 0;
    for (int i = 0; i < num_samples; i++) {
        if (true_labels[i] == predicted_labels[i]) {
            correct++;
        }
    }
    return (float)correct / num_samples;
}

int main() {
    Sample *data;
    int num_samples, num_features;

    readCSV("data.csv", &data, &num_samples, &num_features);

    Sample *train_data, *test_data;
    int num_train, num_test;

    splitDataset(data, num_samples, &train_data, &num_train, &test_data, &num_test, 0.7f);

    float *distances;
    computeDistances(train_data, num_train, test_data, num_test, num_features, &distances);

    int *predicted_labels = (int *)malloc(num_test * sizeof(int));
    CHECK_ERROR(!predicted_labels, "Error allocating memory");

    int *true_labels = (int *)malloc(num_test * sizeof(int));
    CHECK_ERROR(!true_labels, "Error allocating memory");

    for (int i = 0; i < num_test; i++) {
        true_labels[i] = test_data[i].label;
    }

    FILE *output_file = fopen("knn_accuracy.csv", "w");
    CHECK_ERROR(!output_file, "Error opening file");

    fprintf(output_file, "K,Accuracy\n");

    for (int K = 1; K <= 20; K++) {
        classify(distances, train_data, num_train, num_test, K, predicted_labels);
        float accuracy = computeAccuracy(true_labels, predicted_labels, num_test);
        fprintf(output_file, "%d,%.2f\n", K, accuracy * 100);
        printf("K=%d, Accuracy=%.2f%%\n", K, accuracy * 100);
    }

    fclose(output_file);

    free(distances);
    free(predicted_labels);
    free(true_labels);
    for (int i = 0; i < num_samples; i++) {
        free(data[i].data);
    }
    free(data);
    free(train_data);
    free(test_data);

    return 0;
}
