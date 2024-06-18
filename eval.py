#print feature importance of the model
import joblib

import helpers



# sort feature importance map by value
#sorted_feature_importances_map = dict(sorted(feature_importances_map.items(), key=lambda item: item[1], reverse=True))
most_important_features = helpers.get_most_important_features_names(10)

print("sorted: ",most_important_features)


# get the index of a feature by its name
feature_index = helpers.get_feature_index_from_name(most_important_features[0])
print(feature_index)

top_n_feature_importance_indices = [helpers.get_feature_index_from_name(feature) for feature in most_important_features]
print(top_n_feature_importance_indices)
