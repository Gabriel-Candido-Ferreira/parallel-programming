import 'dart:math';
import 'dart:io';

void main() {
  // Criando um vetor com 10.000.000 posições, contendo valores aleatórios entre 0 e 10.
  List<double> notas =
      List.generate(100000000, (index) => Random().nextDouble() * 10);

  // Definindo os intervalos do histograma.
  List<double> intervalos = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];

  // Inicializando o histograma.
  List<int> histograma = List.filled(10, 0);

  // Iniciando o cronômetro para medir o tempo de processamento.
  Stopwatch cronometro = Stopwatch()..start();

  // Preenchendo o histograma.
  for (double nota in notas) {
    for (int i = 0; i < intervalos.length; i++) {
      if (nota <= intervalos[i]) {
        histograma[i]++;
        break;
      }
    }
  }
  // Parando o cronômetro.
  cronometro.stop();

  // Exibindo o resultado.
  print("Histograma:");
  for (int i = 0; i < histograma.length; i++) {
    double min = i == 0 ? 0 : intervalos[i - 1] + 0.1;
    double max = intervalos[i];
    print(
        "De ${min.toStringAsFixed(1)} a ${max.toStringAsFixed(1)}: ${histograma[i]}");
  }

  // Soma das faixas.
  int somaFaixas = histograma.reduce((a, b) => a + b);
  print("Soma das faixas: $somaFaixas");

  // Tempo de processamento em segundos.
  print(
      "Tempo de processamento: ${(cronometro.elapsedMilliseconds / 1000).toStringAsFixed(5)} segundos");
}
