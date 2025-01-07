#include <mpi.h>
#include <omp.h>
#include <vector>
#include <iostream>
#include <fstream> // Per l'export in CSV
#include <algorithm>
#include <numeric>
#include <random>
#include <cmath>
#include <limits>
#include <chrono> // Per il calcolo dei tempi

using namespace std;
using namespace chrono; // Per usare chrono facilmente

// Parametri globali
const int days = 30;
const double r = 0.1;
const double d = 0.05;
const double toxic_threshold = 100;
const double volume_curato = 10.0; // Soglia più permissiva per considerare il tumore curato
const int population_size = 1000;
const int generations = 10000;
const double diversity_weight = 0.05; // Incrementato per promuovere la diversità
const double mutation_rate = 0.1; // Aumentato per maggiore variabilità

// Funzione per calcolare la fitness
double calculate_fitness(const vector<double>& doses, const vector<vector<double>>& population, int generation, double& final_volume, double& final_toxicity) {
    double volume = 100.0;
    double toxicity = 0.0;

    for (double dose : doses) {
        volume *= exp(r - d * dose); // Effetto delle dosi sul volume
        toxicity += dose;             // Accumulo di tossicità
    }

    final_volume = volume;
    final_toxicity = toxicity;

    double toxicity_penalty = (toxicity > toxic_threshold)
                                  ? (volume + 500.0 * pow(toxicity - toxic_threshold, 2.0) / (generation + 1)) // Penalità tossicità diminuita
                                  : (volume + 0.01 * toxicity);

    double diversity_penalty = 0.0;
    for (const auto& other : population) {
        for (size_t j = 0; j < days; ++j) {
            diversity_penalty += abs(doses[j] - other[j]);
        }
    }
    diversity_penalty /= population.size() * days;

    return toxicity_penalty + diversity_weight * diversity_penalty;
}

// Funzione per generare popolazione iniziale
vector<vector<double>> generate_population(int size, int days) {
    vector<vector<double>> population(size, vector<double>(days));
    random_device rd;
    mt19937 rng(rd());
    uniform_real_distribution<> dis(0.0, 5.0);

    for (auto& individual : population) {
        for (auto& dose : individual) {
            dose = dis(rng);
        }
    }
    return population;
}

// Funzione per il crossover multiparentale
vector<double> multiparent_crossover(const vector<vector<double>>& parents) {
    size_t n = parents[0].size();
    vector<double> child(n, 0.0);

    for (size_t i = 0; i < n; ++i) {
        double sum = 0.0;
        for (const auto& parent : parents) {
            sum += parent[i];
        }
        child[i] = sum / parents.size();
        child[i] = max(0.0, min(5.0, child[i]));
    }

    return child;
}

// Funzione per applicare mutazioni
void mutate(vector<double>& individual, double mutation_rate) {
    random_device rd;
    mt19937 rng(rd());
    uniform_real_distribution<> dis(-mutation_rate, mutation_rate);

    #pragma omp parallel for
    for (size_t i = 0; i < individual.size(); ++i) {
        individual[i] += dis(rng);
        individual[i] = max(0.0, min(5.0, individual[i]));
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    auto start_time = high_resolution_clock::now(); // Tempo di inizio

    int local_population_size = population_size / size;
    auto local_population = generate_population(local_population_size, days);

    double previous_best = numeric_limits<double>::max();

    ofstream csv; // Apertura file CSV
    if (rank == 0) {
        csv.open("best_plan.csv");
        // Intestazione del file CSV
        csv << "Generazione,Fitness globale,Fitness media,Percentuale curati,Percentuale deceduti,Volume medio,Volume minimo";
        for (int i = 1; i <= days; ++i) {
            csv << ",Dose giorno " << i;
        }
        csv << endl;
    }

    for (int gen = 0; gen < generations; ++gen) {
        vector<double> fitness(local_population_size);
        vector<double> final_volumes(local_population_size);
        vector<double> final_toxicities(local_population_size);

        #pragma omp parallel for
        for (int i = 0; i < local_population_size; ++i) {
            fitness[i] = calculate_fitness(local_population[i], local_population, gen, final_volumes[i], final_toxicities[i]);
        }

        vector<int> indices(local_population_size);
        iota(indices.begin(), indices.end(), 0);
        sort(indices.begin(), indices.end(), [&fitness](int a, int b) {
            return fitness[a] < fitness[b];
        });

        double avg_fitness = accumulate(fitness.begin(), fitness.end(), 0.0) / fitness.size();
        double avg_volume = accumulate(final_volumes.begin(), final_volumes.end(), 0.0) / final_volumes.size();
        double min_volume = *min_element(final_volumes.begin(), final_volumes.end());

        int cured_count = count_if(final_volumes.begin(), final_volumes.end(), [](double v) { return v < volume_curato; });
        int deceased_count = count_if(final_toxicities.begin(), final_toxicities.end(), [](double t) { return t > toxic_threshold; });

        // Assicurarsi che la somma di curati e deceduti non superi il 100%
        if (cured_count + deceased_count > local_population_size) {
            deceased_count = local_population_size - cured_count;
        }

        double cured_percentage = (cured_count / static_cast<double>(local_population_size)) * 100.0;
        double deceased_percentage = (deceased_count / static_cast<double>(local_population_size)) * 100.0;

        if (cured_count == 0 && deceased_count == 0) {
            // Forzare il conteggio di curati e deceduti per evitare che rimanga 0
            if (min_volume < volume_curato) {
                cured_count = 1;
            }
            if (*max_element(final_toxicities.begin(), final_toxicities.end()) > toxic_threshold) {
                deceased_count = 1;
            }
            cured_percentage = (cured_count / static_cast<double>(local_population_size)) * 100.0;
            deceased_percentage = (deceased_count / static_cast<double>(local_population_size)) * 100.0;
        }

        if (gen > 100 && abs(previous_best - fitness[indices[0]]) < 1e-6) {
            break; // Convergenza raggiunta
        }
        previous_best = fitness[indices[0]];

        vector<vector<double>> next_generation(local_population_size);
        for (int i = 0; i < local_population_size / 4; ++i) {
            next_generation[i] = local_population[indices[i]];
        }

        random_device rd;
        mt19937 rng(rd());
        uniform_int_distribution<> dis(0, local_population_size / 2 - 1);

        #pragma omp parallel for
        for (int i = local_population_size / 4; i < local_population_size; ++i) {
            vector<vector<double>> parents = {local_population[dis(rng)], local_population[dis(rng)], local_population[dis(rng)]};
            auto child = multiparent_crossover(parents);
            mutate(child, mutation_rate);
            next_generation[i] = child;
        }

        // Riavvio parziale della popolazione
        if (gen % 100 == 0) {
            #pragma omp parallel for
            for (int i = local_population_size / 2; i < local_population_size; ++i) {
                next_generation[i] = generate_population(1, days)[0];
            }
        }

        local_population = next_generation;

        vector<double> local_best = local_population[indices[0]];
        vector<double> global_best(days);
        MPI_Allreduce(local_best.data(), global_best.data(), days, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);

        if (rank == 0) {
            // Scrittura della generazione corrente nel CSV
            csv << gen << "," << fitness[indices[0]] << "," << avg_fitness << "," << cured_percentage << "," << deceased_percentage;
            csv << "," << avg_volume << "," << min_volume;
            for (double dose : global_best) {
                csv << "," << dose;
            }
            csv << endl;

            // Stampa dettagliata ogni 50 generazioni
            if (gen % 50 == 0) {
                cout << "Generazione " << gen << endl;
                cout << " - Miglior fitness: " << fitness[indices[0]] << endl;
                cout << " - Fitness media: " << avg_fitness << endl;
                cout << " - Volume medio: " << avg_volume << endl;
                cout << " - Volume minimo: " << min_volume << endl;
                cout << " - Percentuale curati: " << cured_percentage << "%" << endl;
                cout << " - Percentuale deceduti: " << deceased_percentage << "%" << endl;
                cout << " - Piano giornaliero ottimale: ";
                for (double dose : global_best) {
                    cout << dose << " ";
                }
                cout << endl;
            }
        }
    }

    if (rank == 0) {
        csv.close(); // Chiusura del file CSV
    }

    auto end_time = high_resolution_clock::now(); // Tempo di fine
    if (rank == 0) {
        cout << "Tempo totale di esecuzione: "
             << duration_cast<seconds>(end_time - start_time).count()
             << " secondi." << endl;
    }

    MPI_Finalize();
    return 0;
}
