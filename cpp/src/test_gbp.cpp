#include <iostream>
#include <Eigen/Dense>
#include "gbp/FactorGraph.h"

using namespace gbp;

static void printVec(const Eigen::VectorXd& v, const std::string& name) {
    std::cout << name << " = ";
    for (int i = 0; i < v.size(); ++i) std::cout << v[i] << (i+1==v.size()? "" : " ");
    std::cout << "\n";
}

int main() {
    FactorGraph g;
    g.eta_damping = 0.0;

    // Two 2D variables
    auto* x0 = g.addVariable(0, 2);
    auto* x1 = g.addVariable(1, 2);

    // priors: weak (small precision) to avoid singular
    x0->prior.setLam(1e-3 * Eigen::MatrixXd::Identity(2,2));
    x1->prior.setLam(1e-3 * Eigen::MatrixXd::Identity(2,2));
    x0->prior.setEta(Eigen::VectorXd::Zero(2));
    x1->prior.setEta(Eigen::VectorXd::Zero(2));

    // One binary factor between x0 and x1:
    // Encourage x1 - x0 = b with precision W.
    // For a linear constraint (x1 - x0 - b) with W, information form gives:
    // lam = A^T W A, eta = A^T W b, where A = [-I, I]
    Eigen::MatrixXd W = 10.0 * Eigen::MatrixXd::Identity(2,2);
    Eigen::VectorXd b(2); b << 1.0, 2.0;

    auto* f01 = g.addFactor(0, {x0, x1});

    // Connect adjacency with local indices 0 and 1
    g.connect(f01, x0, 0);
    g.connect(f01, x1, 1);

    // Initialize beliefs from priors
    x0->updateBelief();
    x1->updateBelief();
    g.syncAllFactorAdjBeliefs();

    // Set factor info form on concatenated [x0, x1]
    // A = [-I, I] => A^T W A = [[W, -W],[-W, W]]
    Eigen::MatrixXd lam = Eigen::MatrixXd::Zero(4,4);
    lam.block(0,0,2,2) = W;
    lam.block(0,2,2,2) = -W;
    lam.block(2,0,2,2) = -W;
    lam.block(2,2,2,2) = W;

    Eigen::VectorXd eta = Eigen::VectorXd::Zero(4);
    // eta = A^T W b = [-Wb, Wb]
    eta.segment(0,2) = -W*b;
    eta.segment(2,2) =  W*b;

    f01->factor.setLam(lam);
    f01->factor.setEta(eta);

    // Run synchronous iterations with different counts
    std::cout << "Testing different iteration counts:\n\n";
    
    for (int num_iters : {1, 2, 10, 20}) {
        // Reset beliefs
        x0->updateBelief();
        x1->updateBelief();
        g.syncAllFactorAdjBeliefs();
        
        // Run iterations
        for (int it = 0; it < num_iters; ++it) {
            g.synchronousIteration();
        }
        
        std::cout << "After " << num_iters << " synchronous iterations:\n";
        printVec(x0->mu, "  x0.mu");
        printVec(x1->mu, "  x1.mu");
        printVec(x1->mu - x0->mu, "  x1-x0 (should ~ b)");
        std::cout << "\n";
    }
    
    return 0;
}
