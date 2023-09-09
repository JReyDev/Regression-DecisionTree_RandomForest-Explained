#include <iostream>
#include <vector>
#include <Eigen/Dense>
#include <numeric>
#include <limits>

class DecisionNode {
public:
    int feature_index;
    double threshold;
    double value;
    DecisionNode* left;
    DecisionNode* right;

    DecisionNode() : feature_index(0), threshold(0), value(0), left(nullptr), right(nullptr) {}
};

class DecisionTreeRegressor {
public:
    int min_samples_split;
    int max_depth;
    DecisionNode* root;

    DecisionTreeRegressor(int min_samples_split = 2, int max_depth = 2)
        : min_samples_split(min_samples_split), max_depth(max_depth), root(nullptr) {}

    double _cost(const Eigen::VectorXd& y) {
        if (y.size() == 0) {
            return 0;
        }

        double mean = y.mean();
        return (y.array() - mean).square().sum();
    }
    double _split_cost(const Eigen::MatrixXd& X, const Eigen::VectorXd& y, int feature_index, double threshold) {
        
        Eigen::ArrayXd column_values = X.col(feature_index).array();

        Eigen::Array<bool, Eigen::Dynamic, 1> left_indices = column_values < threshold;
        Eigen::Array<bool, Eigen::Dynamic, 1> right_indices = column_values >= threshold;

        std::vector<double> left_values_vector, right_values_vector;

        for (int i = 0; i < y.size(); ++i) {
            if (left_indices(i)) {
                left_values_vector.push_back(y(i));
            }
            if (right_indices(i)) {
                right_values_vector.push_back(y(i));
            }
        }

        Eigen::VectorXd left_values = Eigen::Map<Eigen::VectorXd>(left_values_vector.data(), left_values_vector.size());
        Eigen::VectorXd right_values = Eigen::Map<Eigen::VectorXd>(right_values_vector.data(), right_values_vector.size());


        return _cost(left_values) + _cost(right_values);

    }


    std::pair<int, double> _best_split(const Eigen::MatrixXd& X, const Eigen::VectorXd& y) {

        int best_feature_index = -1;
        double best_threshold = 0;
        double best_cost = std::numeric_limits<double>::infinity();

        for (int feature_index = 0; feature_index < X.cols(); ++feature_index) {

            Eigen::VectorXd unique_values = X.col(feature_index);
            std::sort(unique_values.data(), unique_values.data() + unique_values.size());

            for (int i = 0; i < unique_values.size(); ++i) {

                double threshold = unique_values[i];
                double cost = _split_cost(X, y, feature_index, threshold);

                if (cost < best_cost) {

                    best_cost = cost;
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
            node->value = y.mean();
            return;
        }

        auto [best_feature_index, best_threshold] = _best_split(X, y);

        if (best_feature_index == -1) {
            node->value = y.mean();
            return;
        }

        node->feature_index = best_feature_index;
        node->threshold = best_threshold;

        Eigen::ArrayXd column_values = X.col(best_feature_index).array();
        Eigen::Array<bool, Eigen::Dynamic, 1> left_indices = column_values < best_threshold;
        Eigen::Array<bool, Eigen::Dynamic, 1> right_indices = column_values >= best_threshold;

        std::vector<double> y_left, y_right;

        for (int i = 0; i < y.size(); ++i) {
            if (left_indices(i)) {
                y_left.push_back(y(i));
            }
            if (right_indices(i)) {
                y_right.push_back(y(i));
            }
        }

        Eigen::VectorXd left_values = Eigen::Map<Eigen::VectorXd>(y_left.data(), y_left.size());
        Eigen::VectorXd right_values = Eigen::Map<Eigen::VectorXd>(y_right.data(), y_right.size());

        node->left = new DecisionNode();
        node->right = new DecisionNode();

        fit(X, left_values, node->left, depth + 1);
        fit(X, right_values, node->right, depth + 1);
    }

    double predict(const Eigen::VectorXd& X, DecisionNode* node = nullptr) {
        if (node == nullptr) {
            node = root;
        }

        if (node->value != 0) {
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

double _cost_t(const Eigen::VectorXd& y) {
    if (y.size() == 0) {
        return 0;
    }

    double mean = y.mean();
    return (y.array() - mean).square().sum();
}

double _split_cost_t(const Eigen::MatrixXd& X, const Eigen::VectorXd& y, int feature_index, double threshold) {

    Eigen::ArrayXd column_values = X.col(feature_index).array();

    Eigen::Array<bool, Eigen::Dynamic, 1> left_indices = column_values < threshold;
    Eigen::Array<bool, Eigen::Dynamic, 1> right_indices = column_values >= threshold;

    std::vector<double> left_values_vector, right_values_vector;

    for (int i = 0; i < y.size(); ++i) {
        if (left_indices(i)) {
            left_values_vector.push_back(y(i));
        }
        if (right_indices(i)) {
            right_values_vector.push_back(y(i));
        }
    }

    Eigen::VectorXd left_values = Eigen::Map<Eigen::VectorXd>(left_values_vector.data(), left_values_vector.size());
    Eigen::VectorXd right_values = Eigen::Map<Eigen::VectorXd>(right_values_vector.data(), right_values_vector.size());


    return _cost_t(left_values) + _cost_t(right_values);

}

std::pair<int, double> _best_split_t(const Eigen::MatrixXd& X, const Eigen::VectorXd& y) {
    int best_feature_index = -1;
    double best_threshold = 0;
    double best_cost = std::numeric_limits<double>::infinity();

    for (int feature_index = 0; feature_index < X.cols(); ++feature_index) {
        Eigen::VectorXd unique_values = X.col(feature_index);
        std::sort(unique_values.data(), unique_values.data() + unique_values.size());

        for (int i = 0; i < unique_values.size(); ++i) {
            double threshold = unique_values[i];
            double cost = _split_cost_t(X, y, feature_index, threshold);

            if (cost < best_cost) {
                best_cost = cost;
                best_feature_index = feature_index;
                best_threshold = threshold;
            }
        }
    }

    return { best_feature_index, best_threshold };
}

int main() {

    // Trading volume (in '000s)
    Eigen::VectorXd volume(8);
    volume << 1000, 1200, 1500, 2000, 1800, 2100, 1900, 1600;

    // Opening price (in $)
    Eigen::VectorXd open_price(8);
    open_price << 50, 52, 51, 53, 54, 56, 55, 57;

    // Closing price (in $)
    Eigen::VectorXd close_price(8);
    close_price << 51.49671415, 53.0617357, 53.14768854, 56.52302986, 55.56584663, 57.86586304, 58.47921282, 59.36743473;

    // Features
    Eigen::MatrixXd X(8, 2);
    X.col(0) = volume;
    X.col(1) = open_price;

    // Target
    Eigen::VectorXd y = close_price;

    // Create and train the decision tree
    
    DecisionTreeRegressor tree(2,5);
    tree.fit(X, y);

    // You can now use the 'tree' object for predictions
    Eigen::VectorXd sample(2);
    sample << 1500, 51;
    double prediction = tree.predict(sample);

    std::cout << "Prediction for sample: " << prediction << std::endl;


    //Testing

    std::cout << "cost: " << _cost_t(close_price) << std::endl;
    std::cout << "split cost " << _split_cost_t(X, y, 0, 1500) << std::endl;

    auto [value1, value2] = _best_split_t(X, y);
    std::cout << "best_feature_index: " << value1 << ", best_threshold: " << value2 << std::endl;


    return 0;
}