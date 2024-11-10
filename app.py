from src.models import AnomalyConv2DDetector, AnomalyDenseDetector, AnomalyLeakyDetector
from src.dataframe import get_database_for_training
from src.training import get_fitness
from src.report import generate_report
from src.constants import f_small_samples

x_train, x_test, y_train, y_test, f_train, f_test = get_database_for_training(f_small_samples)

autoencoderConv2D = AnomalyConv2DDetector()
history = get_fitness(autoencoderConv2D, 'small_conv2D', x_train, x_test)
generate_report('small_conv2D', history, autoencoderConv2D, x_test, y_test, f_test)

autoencoderDense = AnomalyDenseDetector()
history = get_fitness(autoencoderDense, 'small_dense', x_train, x_test)
generate_report('small_dense', history, autoencoderDense, x_test, y_test, f_test)

autoencoderDense = AnomalyLeakyDetector()
history = get_fitness(autoencoderDense, 'small_leak', x_train, x_test)
generate_report('small_leak', history, autoencoderDense, x_test, y_test, f_test)
