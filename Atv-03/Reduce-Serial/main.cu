#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>

int reduce_serial(const std::vector<int>& array, int& trabalho, int& passos) {
    int soma = 0;
    for (int valor : array) {
        soma += valor;
        trabalho++;  // Incrementa o trabalho a cada adição
    }
    passos = array.size();  // Cada elemento conta como um passo
    return soma;
}

void medir_tempo(int tamanho) {
    std::vector<int> array(tamanho, 1);
    int trabalho = 0, passos = 0;

    auto inicio = std::chrono::high_resolution_clock::now();
    int resultado = reduce_serial(array, trabalho, passos);
    auto fim = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> duracao = fim - inicio;

    std::cout << std::fixed << std::setprecision(6);
    std::cout << "Tamanho do array: " << tamanho
              << " | Tempo: " << duracao.count() << " segundos"
              << " | Trabalho: " << trabalho
              << " | Passos: " << passos << std::endl;
}

int main() {
    int tamanhos[] = {100, 1000, 10000, 100000, 1000000, 10000000};
    for (int tamanho : tamanhos) {
        medir_tempo(tamanho);
    }
    return 0;
}