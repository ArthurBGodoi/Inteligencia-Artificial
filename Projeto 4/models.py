import nn
import numpy as np

class PerceptronModel(object):

    def __init__(self, dimensions):

        self.w = nn.Parameter(1, dimensions)

    def get_weights(self):

        return self.w

    def run(self, x):

        return nn.DotProduct(x, self.w)

    def get_prediction(self, x):

        pontuacao = self.run(x)
        valores = nn.as_scalar(pontuacao)

        return 1 if valores >= 0 else -1

    def train(self, dataset):

        convergido = False

        while not convergido:
            convergido = True

            for x, y in dataset.iterate_once(1):

                valor_y = nn.as_scalar(y)  
                predicao = self.get_prediction(x)

                if predicao != valor_y:

                    dado_x = x.data
                    direcao = nn.Constant(valor_y * dado_x) 
                    self.w.update(direcao, 1.0)
                    convergido = False


class RegressionModel(object):

    def __init__(self):

        valor_1 = 128
        valor_2 = 64
        valor_3 = 32

        self.w1 = nn.Parameter(1, valor_1)
        self.b1 = nn.Parameter(1, valor_1)
        self.w2 = nn.Parameter(valor_1, valor_2)
        self.b2 = nn.Parameter(1, valor_2)
        self.w3 = nn.Parameter(valor_2, valor_3)
        self.b3 = nn.Parameter(1, valor_3)
        self.w4 = nn.Parameter(valor_3, 1)
        self.b4 = nn.Parameter(1, 1)

    def run(self, x):

        v1 = nn.AddBias(nn.Linear(x, self.w1), self.b1)
        v1_act = nn.ReLU(v1)
        
        v2 = nn.AddBias(nn.Linear(v1_act, self.w2), self.b2)
        v2_act = nn.ReLU(v2)
        
        v3 = nn.AddBias(nn.Linear(v2_act, self.w3), self.b3)
        v3_act = nn.ReLU(v3)
        
        saida = nn.AddBias(nn.Linear(v3_act, self.w4), self.b4)
        
        return saida

    def get_loss(self, x, y):

        predicao = self.run(x)

        return nn.SquareLoss(predicao, y)

    def train(self, dataset):

        tamanho_lote = 50 
        taxa_aprendizado = 0.01
        parametros = [self.w1, self.b1, self.w2, self.b2, self.w3, self.b3, self.w4, self.b4]
        
        melhor_perda = float('inf')
        paciencia = 200
        contador_paciencia = 0
        
        for epoca in range(5000):
            perda_total = 0
            contador = 0
            
            for x, y in dataset.iterate_once(tamanho_lote):
                perda = self.get_loss(x, y)
                valor_perda = nn.as_scalar(perda)
                perda_total += valor_perda
                contador += 1
                
                # Calcular gradientes e atualizar parâmetros
                gradientes = nn.gradients(perda, parametros)
                for parametro, gradiente in zip(parametros, gradientes):
                    parametro.update(gradiente, -taxa_aprendizado)
            
            perda_media = perda_total / contador
            
            # Decaimento da taxa de aprendizado
            if epoca > 0 and epoca % 500 == 0:
                taxa_aprendizado *= 0.8
            
            # Parada antecipada
            if perda_media < melhor_perda:
                melhor_perda = perda_media
                contador_paciencia = 0
            else:
                contador_paciencia += 1
            
            # Parar quando atingir a meta ou ficar estagnado
            if perda_media < 0.015 or contador_paciencia >= paciencia:
                break


class DigitClassificationModel(object):

    def __init__(self):
        
        valor = 256

        self.w1 = nn.Parameter(784, valor)
        self.b1 = nn.Parameter(1, valor)
        self.w2 = nn.Parameter(valor, 10)
        self.b2 = nn.Parameter(1, 10)

    def run(self, x):

        valor = nn.AddBias(nn.Linear(x, self.w1), self.b1)
        valor_relu = nn.ReLU(valor)
        logits = nn.AddBias(nn.Linear(valor_relu, self.w2), self.b2)

        return logits

    def get_loss(self, x, y):

        logits = self.run(x)
        return nn.SoftmaxLoss(logits, y)

    def train(self, dataset):

        taxa_inicial = 0.1
        taxa_aprendizado = taxa_inicial
        tamanho_lote = 100
        parametros = [self.w1, self.b1, self.w2, self.b2]

        maximo_epocas = 100  
        melhor_acc_validacao = 0
        paciencia = 10  
        contador_paciencia = 0
        
        for epoca in range(maximo_epocas):

            for x, y in dataset.iterate_once(tamanho_lote):
                perda = self.get_loss(x, y)
                gradientes = nn.gradients(perda, parametros)

                for parametro, gradiente in zip(parametros, gradientes):
                    parametro.update(gradiente, -taxa_aprendizado)
            
            # Decaimento da taxa de aprendizado
            if epoca > 0 and epoca % 20 == 0:
                taxa_aprendizado = taxa_inicial * (0.5 ** (epoca // 20))
            
            # Verificação de validação
            if hasattr(dataset, "get_validation_accuracy"):
                acc_validacao = dataset.get_validation_accuracy()
                
                if acc_validacao > melhor_acc_validacao:
                    melhor_acc_validacao = acc_validacao
                    contador_paciencia = 0
                else:
                    contador_paciencia += 1
                
                # Parar se atingiu a meta OU se não melhora há muito tempo
                if acc_validacao >= 0.975 or contador_paciencia >= paciencia:
                    break
            else:
                # Sem validação, treinar por épocas fixas
                if epoca >= 30:  # Mínimo de épocas
                    break



class LanguageIDModel(object):

    def __init__(self):

        self.num_chars = 47
        self.languages = ["English", "Spanish", "Finnish", "Dutch", "Polish"]

        tamanho_oculto = 256 

        self.w_xh = nn.Parameter(self.num_chars, tamanho_oculto)
        self.w_hh = nn.Parameter(tamanho_oculto, tamanho_oculto)
        self.b_h = nn.Parameter(1, tamanho_oculto)
        self.w_hy = nn.Parameter(tamanho_oculto, len(self.languages))
        self.b_y = nn.Parameter(1, len(self.languages))

    def run(self, xs):

        tamanho_lote = xs[0].data.shape[0]
        tamanho_oculto = self.w_hh.data.shape[0]

        h = nn.Constant(np.zeros((tamanho_lote, tamanho_oculto)))

        for x in xs:
            x_para_h = nn.Linear(x, self.w_xh)
            h_para_h = nn.Linear(h, self.w_hh)
            pre_ativacao = nn.Add(x_para_h, h_para_h)
            pre_ativacao = nn.AddBias(pre_ativacao, self.b_h)
            h = nn.ReLU(pre_ativacao)

        logits = nn.AddBias(nn.Linear(h, self.w_hy), self.b_y)
        return logits

    def get_loss(self, xs, y):

        logits = self.run(xs)
        return nn.SoftmaxLoss(logits, y)

    def train(self, dataset):

        taxa_inicial = 0.1
        taxa_aprendizado = taxa_inicial
        tamanho_lote = 100
        parametros = [self.w_xh, self.w_hh, self.b_h, self.w_hy, self.b_y]
        maximo_epocas = 80  
        epoca = 0

        melhor_acc_validacao = 0
        paciencia = 8 
        contador_paciencia = 0

        while epoca < maximo_epocas:
            epoca += 1
            
            # Treinamento
            for xs, y in dataset.iterate_once(tamanho_lote):
                perda = self.get_loss(xs, y)
                gradientes = nn.gradients(perda, parametros)
                for parametro, gradiente in zip(parametros, gradientes):
                    parametro.update(gradiente, -taxa_aprendizado)

            # Validação
            if hasattr(dataset, "get_validation_accuracy"):
                acc_validacao = dataset.get_validation_accuracy()
                
                if epoca > 10 and acc_validacao < melhor_acc_validacao:
                    taxa_aprendizado = taxa_inicial * (0.8 ** (contador_paciencia // 2))
                
                if acc_validacao > melhor_acc_validacao:
                    melhor_acc_validacao = acc_validacao
                    contador_paciencia = 0
                else:
                    contador_paciencia += 1
                
                if acc_validacao >= 0.86 or contador_paciencia >= paciencia:
                    break
                    
            else:
                if epoca >= maximo_epocas:
                    break