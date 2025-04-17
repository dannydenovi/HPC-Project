#include <omp.h>
#include <mpi.h>
#include <vector>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <numeric>
#include <random>
#include <cmath>
#include <limits>
#include <chrono>

using namespace std;
using namespace chrono;

// Parametri globali
const int days = 30;
const double r = 0.09;
const double d = 0.05;
const double toxic_threshold = 100;
const double volume_curato = 10.0;
const int population_size = 1000;
const int generations = 2500;
const double diversity_weight = 0.05;
const double mutation_rate = 0.1;

// Funzione fitness
double calculate_fitness(const vector<double>& doses, const vector<vector<double>>& population, int generation, double& final_volume, double& final_toxicity) {
    double volume = 100.0;
    double toxicity = 0.0;
    for (double dose : doses) {
        volume *= exp(r - d * dose);
        toxicity += dose;
    }
    final_volume = volume;
    final_toxicity = toxicity;
    double toxicity_penalty = (toxicity > toxic_threshold)
        ? (volume + 500.0 * pow(toxicity - toxic_threshold, 2.0) / (generation + 1))
        : (volume + 0.01 * toxicity);
    double diversity_penalty = 0.0;
    for (const auto& other : population)
        for (size_t j = 0; j < days; ++j)
            diversity_penalty += abs(doses[j] - other[j]);
    diversity_penalty /= population.size() * days;
    return toxicity_penalty + diversity_weight * diversity_penalty;
}

vector<vector<double>> generate_population(int size, int days) {
    vector<vector<double>> population(size, vector<double>(days));
    random_device rd;
    mt19937 rng(rd());
    uniform_real_distribution<> dis(0.0, 5.0);
    for (auto& individual : population)
        for (auto& dose : individual)
            dose = dis(rng);
    return population;
}

vector<double> multiparent_crossover(const vector<vector<double>>& parents) {
    size_t n = parents[0].size();
    vector<double> child(n, 0.0);
    for (size_t i = 0; i < n; ++i) {
        double sum = 0.0;
        for (const auto& parent : parents) sum += parent[i];
        child[i] = max(0.0, min(5.0, sum / parents.size()));
    }
    return child;
}

void mutate(vector<double>& individual, double mutation_rate) {
    random_device rd;
    mt19937 rng(rd());
    uniform_real_distribution<> dis(-mutation_rate, mutation_rate);
    for (size_t i = 0; i < individual.size(); ++i) {
        individual[i] += dis(rng);
        individual[i] = max(0.0, min(5.0, individual[i]));
    }
}

int main(int argc, char** argv) {
    int omp_threads = 0;
    int mpi_procs = 0;

    // Parsing CLI args
    for (int i = 1; i < argc; ++i) {
        if (string(argv[i]) == "--omp-threads" && i + 1 < argc) {
            omp_threads = stoi(argv[++i]);
        }
        if (string(argv[i]) == "--mpi-procs" && i + 1 < argc) {
            mpi_procs = stoi(argv[++i]);
        }
    }
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Imposta 2 thread per core dopo l'inizializzazione di MPI
    int num_cores = omp_get_num_procs();
    if (omp_threads > 0) {
        omp_set_num_threads(omp_threads);
    } else {
        omp_set_num_threads(2 * num_cores); // fallback default
    }

    if (rank == 0) {
        cout << "Thread per core: 2" << endl;
        cout << "Totale thread utilizzati: " << omp_get_max_threads() << endl;
        cout << "Totale processi MPI: " << size << endl;
        cout << "Totale unità di parallelismo (thread * core): " << (omp_get_max_threads() * size) << endl;
        cout << "Totale thread utilizzati: " << omp_get_max_threads() << endl;
    }

    auto start_time = high_resolution_clock::now();

    int local_population_size = population_size / size;
    vector<vector<double>> pop_buffer[2];
    vector<double> fitness_buffer[2];
    vector<double> final_volumes[2];
    vector<double> final_toxicities[2];

    pop_buffer[0] = generate_population(local_population_size, days);
    fitness_buffer[0].resize(local_population_size);
    final_volumes[0].resize(local_population_size);
    final_toxicities[0].resize(local_population_size);

    ofstream csv;
    if (rank == 0) {
        csv.open("pipeline_result.csv");
        csv << "Gen,Fitness_avg,Volume_avg,Cured_percent,Deceased_percent\n";
    }

    #pragma omp parallel
    {
        #pragma omp single
        {
            for (int gen = 0; gen < generations; ++gen) {
                int curr = gen % 2;
                int next = (gen + 1) % 2;

                pop_buffer[next].resize(local_population_size);
                for (auto& individual : pop_buffer[next])
                    individual.resize(days);
                fitness_buffer[next].resize(local_population_size);
                final_volumes[next].resize(local_population_size);
                final_toxicities[next].resize(local_population_size);

                double* fitness_ptr = fitness_buffer[curr].data();
                double* volume_ptr = final_volumes[curr].data();
                double* tox_ptr = final_toxicities[curr].data();

                #pragma omp task depend(out: fitness_ptr[0:local_population_size])
                {
                    for (int i = 0; i < local_population_size; ++i)
                        fitness_ptr[i] = calculate_fitness(pop_buffer[curr][i], pop_buffer[curr], gen, volume_ptr[i], tox_ptr[i]);
                }

                #pragma omp task depend(in: fitness_ptr[0:local_population_size])
                {
                    vector<int> indices(local_population_size);
                    iota(indices.begin(), indices.end(), 0);
                    sort(indices.begin(), indices.end(), [&](int a, int b) {
                        return fitness_ptr[a] < fitness_ptr[b];
                    });
                    random_device rd;
                    mt19937 rng(rd());
                    uniform_int_distribution<> dis(0, local_population_size / 2 - 1);

                    for (int i = 0; i < local_population_size / 4; ++i)
                        pop_buffer[next][i] = pop_buffer[curr][indices[i]];

                    for (int i = local_population_size / 4; i < local_population_size; ++i) {
                        vector<vector<double>> parents = {pop_buffer[curr][dis(rng)], pop_buffer[curr][dis(rng)], pop_buffer[curr][dis(rng)]};
                        auto child = multiparent_crossover(parents);
                        mutate(child, mutation_rate);
                        pop_buffer[next][i] = child;
                    }
                }

                if (rank == 0) {
                    #pragma omp task depend(in: fitness_ptr[0:local_population_size])
                    {
                        double avg_fitness = accumulate(fitness_buffer[curr].begin(), fitness_buffer[curr].end(), 0.0) / local_population_size;
                        double avg_volume = accumulate(final_volumes[curr].begin(), final_volumes[curr].end(), 0.0) / local_population_size;
                        int cured_count = count_if(final_volumes[curr].begin(), final_volumes[curr].end(), [](double v) { return v < volume_curato; });
int deceased_count = local_population_size - cured_count;
double cured_percent = (cured_count / static_cast<double>(local_population_size)) * 100.0;
double deceased_percent = 100.0 - cured_percent;
csv << gen << "," << avg_fitness << "," << avg_volume << "," << cured_percent << "," << deceased_percent << "\n";
                    }
                }
            }
        }
    }

    if (rank == 0) {
        csv.close();
        auto end_time = high_resolution_clock::now();
        cout << "Esecuzione completata in "
             << duration_cast<seconds>(end_time - start_time).count()
             << " secondi." << endl;
    }

    MPI_Finalize();
    return 0;
}
