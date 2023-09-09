#include <iostream>
#include <vector>
#include <Eigen/Dense>
#include <map>


class DecisionNode {
public:
    int feature_index; // This remains int because it refers to an index
    double threshold;
    double value;      // Changed to double
    DecisionNode* left;
    DecisionNode* right;

    DecisionNode() : feature_index(-1), threshold(0.0), value(-1.0), left(nullptr), right(nullptr) {}
};

class DecisionTreeClassifier {
public:
    double min_samples_split;  // Changed to double
    double max_depth;          // Changed to double
    DecisionNode* root;

    DecisionTreeClassifier(double min_samples_split = 2.0, double max_depth = 5.0)
        : min_samples_split(min_samples_split), max_depth(max_depth), root(nullptr) {}

    double _gini_index(const Eigen::VectorXd& y) { // Changed to Eigen::VectorXd
        std::map<int, double> label_count;         // Value type changed to double
        for (int i = 0; i < y.size(); ++i) {
            label_count[static_cast<int>(y[i])]++; // Explicit casting to int for map indexing
        }

        double impurity = 1.0;
        for (auto const& pair : label_count) {
            double prob = pair.second / y.size();
            impurity -= prob * prob;
        }
        return impurity;
    }

    std::pair<int, double> _best_split(const Eigen::MatrixXd& X, const Eigen::VectorXd& y) {
        int best_feature_index = -1;
        double best_threshold = 0;
        double best_gini = 1;

        for (int feature_index = 0; feature_index < X.cols(); ++feature_index) {
            Eigen::VectorXd unique_values = X.col(feature_index);
            std::sort(unique_values.data(), unique_values.data() + unique_values.size());

            for (int i = 0; i < unique_values.size(); ++i) {
                double threshold = unique_values[i];
                Eigen::Array<bool, Eigen::Dynamic, 1> left_indices = (X.col(feature_index).array() < threshold);
                Eigen::Array<bool, Eigen::Dynamic, 1> right_indices = (X.col(feature_index).array() >= threshold);

                Eigen::VectorXd y_left = y.array().select(left_indices.cast<double>(), 0);
                Eigen::VectorXd y_right = y.array().select(right_indices.cast<double>(), 0);

                double gini = (y_left.size() / y.size()) * _gini_index(y_left) +
                    (y_right.size() / y.size()) * _gini_index(y_right);

                if (gini < best_gini) {
                    best_gini = gini;
                    best_feature_index = feature_index;
                    best_threshold = threshold;
                }
            }
        }

        return { best_feature_index, best_threshold };
    }

    void fit(const Eigen::MatrixXd& X, const Eigen::VectorXd& y, DecisionNode* node = nullptr, int depth = 0) {
        if (node == nullptr) {
            node = new DecisionNode();
            root = node;
        }

        if (y.size() < min_samples_split || depth == max_depth) {
            node->value = *(std::max_element(y.data(), y.data() + y.size()));
            return;
        }

        auto [best_feature_index, best_threshold] = _best_split(X, y);

        if (best_feature_index == -1) {
            node->value = *(std::max_element(y.data(), y.data() + y.size()));
            return;
        }

        node->feature_index = best_feature_index;
        node->threshold = best_threshold;

        node->left = new DecisionNode();
        node->right = new DecisionNode();

        Eigen::Array<bool, Eigen::Dynamic, 1> left_indices = (X.col(best_feature_index).array() < best_threshold);
        Eigen::Array<bool, Eigen::Dynamic, 1> right_indices = (X.col(best_feature_index).array() >= best_threshold);

        Eigen::VectorXd y_left = y.array().select(left_indices.cast<double>(), 0);
        Eigen::VectorXd y_right = y.array().select(right_indices.cast<double>(), 0);

        fit(X, y_left, node->left, depth + 1);
        fit(X, y_right, node->right, depth + 1);
    }

    double predict(const Eigen::VectorXd& X, DecisionNode* node = nullptr) {
        if (node == nullptr) {
            node = root;
        }

        if (node->value != -1.0) {  // Changed comparison value to -1.0
            return node->value;
        }

        if (X(node->feature_index) < node->threshold) {
            return predict(X, node->left);
        }
        else {
            return predict(X, node->right);
        }
    }
};

struct Data {
    std::string weather;
    int temperature;
    std::string goOutside;
};

class DataVectorizer {
private:
    std::map<std::string, int> weather_map_;
    std::map<std::string, int> goOutside_map_;
    int num_features_ = 3;  // weather, temperature, goOutside

public:
    void fit(const std::vector<Data>& data) {
        for (const auto& entry : data) {
            if (weather_map_.find(entry.weather) == weather_map_.end()) {
                int next_index = static_cast<int>(weather_map_.size());
                weather_map_[entry.weather] = next_index;
            }

            if (goOutside_map_.find(entry.goOutside) == goOutside_map_.end()) {
                int next_index = static_cast<int>(goOutside_map_.size());
                goOutside_map_[entry.goOutside] = next_index;
            }
        }
    }

    Eigen::MatrixXd transform(const std::vector<Data>& data) {
        Eigen::MatrixXd X(data.size(), num_features_);

        int row = 0;
        for (const auto& entry : data) {
            X(row, 0) = static_cast<double>(weather_map_[entry.weather]);
            X(row, 1) = static_cast<double>(entry.temperature);
            X(row, 2) = static_cast<double>(goOutside_map_[entry.goOutside]);
            row++;
        }

        return X;
    }
};



int main() {

    //Weather, Temperature, Go Outside?

    std::vector<Data> data = {
        {"Cloudy", 21, "Yes"},
        {"Sunny", 24, "Yes"},
        {"Sunny", 26, "Yes"},
        {"Rainy", 19, "No"},
        {"Cloudy", 22, "Yes"},
        {"Rainy", 18, "No"},
        {"Sunny", 27, "Yes"},
        {"Cloudy", 21, "Yes"},
        {"Rainy", 20, "No"},
        {"Sunny", 25, "Yes"}
    };

    

    DataVectorizer vectorizer;
    vectorizer.fit(data);
    Eigen::MatrixXd X_vec = vectorizer.transform(data);

    Eigen::MatrixXd X = X_vec.block(0, 0, X_vec.rows(), 2);
    Eigen::VectorXd y = X_vec.col(2);

    DecisionTreeClassifier TreeClassifier;

    TreeClassifier.fit(X,y);

    Eigen::VectorXd X_test(2);
    X_test << 2, 18;

    std::cout << TreeClassifier.predict(X_test);

    return 0;
}