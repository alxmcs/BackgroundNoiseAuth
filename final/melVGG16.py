from specrogramPreprocessing import spectrogramm_size_preprocessing, contrast_streching
import distances as d
import pandas
import display
from baseFunctions import define_class, get_data_set
from netBuilding import siam_net, sub_net

model1 = sub_net()
model2 = sub_net()

data_set, files, srs = get_data_set("dataset")
test_len = round(len(data_set))

euclideans = []
manhettens = []
chebysheves = []
cosinuses = []
pairs = []

for i in range(test_len):
    for j in range(i+1, test_len):

        mel1 = display.audio_to_mel(data_set[i], srs[i])
        mel2 = display.audio_to_mel(data_set[j], srs[j])

        mel1 = mel1[0:127]
        mel2 = mel2[0:127]

        pair = define_class(files[i], files[j])

        first_transformed = contrast_streching(spectrogramm_size_preprocessing(mel1))
        second_transformed = contrast_streching(spectrogramm_size_preprocessing(mel2))

        first_vect, second_vect = siam_net(model1, model2, first_transformed, second_transformed)

        r1 = d.euclidean_dist(first_vect, second_vect)
        r2 = d.dist_manhetten(first_vect, second_vect)
        r3 = d.dist_cos(first_vect, second_vect)
        r4 = d.dist_chebyshev(first_vect, second_vect)

        euclideans.append(r1)
        manhettens.append(r2)
        cosinuses.append(r3)
        chebysheves.append(r4)

        pairs.append(pair)



df = pandas.DataFrame({'euclidean': euclideans, 'manhetten': manhettens, 'cosinuses': cosinuses, 'chebysheves':chebysheves, 'maxes': maxes, 'is a pair': pairs})
df.to_csv('mel_metrics.csv')
