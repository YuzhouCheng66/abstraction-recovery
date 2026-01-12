#include "gbp/Factor.h"
#include "gbp/VariableNode.h"

namespace gbp {

Factor::Factor(int id, const std::vector<VariableNode*>& vars)
    : id(id),
      factorID(id),
      adj_var_nodes(vars),
      var_nodes(vars) {
    assert(adj_var_nodes.size() == 1 || adj_var_nodes.size() == 2);

    adj_vIDs.reserve(adj_var_nodes.size());
    int total_dofs = 0;
    for (auto* v : adj_var_nodes) {
        assert(v != nullptr);
        adj_vIDs.push_back(v->variableID);
        total_dofs += v->dofs;
    }
    
    dim = total_dofs;
    factor = utils::NdimGaussian(total_dofs);

    adj_beliefs.reserve(adj_var_nodes.size());
    messages.reserve(adj_var_nodes.size());
    for (auto* v : adj_var_nodes) {
        adj_beliefs.emplace_back(v->dofs);
        messages.emplace_back(v->dofs);
    }
}

Factor::Factor()
    : id(-1),
      factorID(-1),
      dim(0),
      factor(0) {}

void Factor::syncAdjBeliefsFromVariables() {
    for (int i = 0; i < (int)adj_var_nodes.size(); ++i) {
        auto* v = adj_var_nodes[i];
        adj_beliefs[i].setEta(v->belief.eta());
        adj_beliefs[i].setLam(v->belief.lam());
    }
}

void Factor::computeFactor() {
    /**
     * Compute factor distribution from measurements and precisions.
     * 
     * For linear SLAM factors:
     * - Unary (prior): constraint is x ≈ z with precision Lambda
     *   J = [I], measurement = z
     *   lambda = I^T @ Lambda @ I = Lambda
     *   eta = I^T @ Lambda @ z = Lambda @ z
     *
     * - Binary (odometry): constraint is (x1 - x0) ≈ z with precision Lambda
     *   J = [-I, I], measurement = z
     *   lambda = [-I^T, I^T] @ Lambda @ [-I, I] = [[L, -L], [-L, L]]
     *   eta = [-I^T, I^T] @ Lambda @ z = [-Lambda @ z; Lambda @ z]
     */
    
    if (adj_var_nodes.size() == 1) {
        // Unary factor: p(x) ∝ exp(-0.5 * (x - z)^T Lambda (x - z))
        // Using canonical form: eta = Lambda * z, lam = Lambda
        assert(measurements.size() >= 1 && precisions.size() >= 1);
        const auto& z = measurements[0];
        const auto& Lambda = precisions[0];
        
        factor.setLam(Lambda);
        factor.setEta(Lambda * z);
        
    } else if (adj_var_nodes.size() == 2) {
        // Binary factor with relative measurement z = x1 - x0
        // Jacobian: J = [-I, I]  (2x4 for 2D case)
        // constraint: (x1 - x0) ≈ z with precision Lambda (2x2)
        // 
        // Precision matrix: J^T @ Lambda @ J = [[-I^T], [I^T]] @ Lambda @ [-I, I]
        //                                     = [[L, -L], [-L, L]]
        // Information vector: J^T @ Lambda @ z = [[-I^T], [I^T]] @ Lambda @ z
        //                                       = [-Lambda @ z; Lambda @ z]
        
        assert(measurements.size() >= 1 && precisions.size() >= 1);
        const auto& z = measurements[0];
        const auto& Lambda = precisions[0];
        
        int d0 = adj_var_nodes[0]->dofs;  // typically 2
        int d1 = adj_var_nodes[1]->dofs;  // typically 2
        
        // Construct precision matrix: [[Lambda, -Lambda], [-Lambda, Lambda]]
        Eigen::MatrixXd lam_factor = Eigen::MatrixXd::Zero(d0 + d1, d0 + d1);
        lam_factor.block(0, 0, d0, d0) = Lambda;           // Upper-left: Lambda
        lam_factor.block(d0, d0, d1, d1) = Lambda;         // Lower-right: Lambda
        lam_factor.block(0, d0, d0, d1) = -Lambda;         // Upper-right: -Lambda
        lam_factor.block(d0, 0, d1, d0) = -Lambda;         // Lower-left: -Lambda
        
        // Construct information vector: [-Lambda @ z; Lambda @ z]
        Eigen::VectorXd eta_factor = Eigen::VectorXd::Zero(d0 + d1);
        eta_factor.segment(0, d0) = -Lambda * z;           // Top block: -Lambda @ z
        eta_factor.segment(d0, d1) = Lambda * z;           // Bottom block: Lambda @ z
        
        factor.setLam(lam_factor);
        factor.setEta(eta_factor);
    }
}

void Factor::computeMessages(double eta_damping) {
    if (!active) return;

    const int n = (int)adj_var_nodes.size();
    if (n == 1) {
        messages[0].setEta(factor.eta());
        messages[0].setLam(factor.lam());
        return;
    }

    VariableNode* v0 = adj_var_nodes[0];
    VariableNode* v1 = adj_var_nodes[1];
    const int d0 = v0->dofs;
    const int d1 = v1->dofs;

    // ---- IMPORTANT: snapshot old messages (Python-style) ----
    const Eigen::VectorXd old_eta0 = messages[0].eta();
    const Eigen::MatrixXd old_lam0 = messages[0].lam();
    const Eigen::VectorXd old_eta1 = messages[1].eta();
    const Eigen::MatrixXd old_lam1 = messages[1].lam();

    Eigen::VectorXd new_eta[2];
    Eigen::MatrixXd new_lam[2];

    for (int target = 0; target < 2; ++target) {
        Eigen::VectorXd eta_f = factor.eta();
        Eigen::MatrixXd lam_f = factor.lam();

        // Multiply in incoming of OTHER variable, but subtract OLD message of OTHER
        if (target == 0) {
            eta_f.segment(d0, d1).noalias() += (adj_beliefs[1].eta() - old_eta1);
            lam_f.block(d0, d0, d1, d1).noalias() += (adj_beliefs[1].lam() - old_lam1);
        } else {
            eta_f.segment(0, d0).noalias() += (adj_beliefs[0].eta() - old_eta0);
            lam_f.block(0, 0, d0, d0).noalias() += (adj_beliefs[0].lam() - old_lam0);
        }

        Eigen::VectorXd eo, eno;
        Eigen::MatrixXd loo, lono, lnoo, lnono;

        if (target == 0) {
            eo = eta_f.segment(0, d0);
            eno = eta_f.segment(d0, d1);
            loo   = lam_f.block(0, 0, d0, d0);
            lono  = lam_f.block(0, d0, d0, d1);
            lnoo  = lam_f.block(d0, 0, d1, d0);
            lnono = lam_f.block(d0, d0, d1, d1);
        } else {
            eo = eta_f.segment(d0, d1);
            eno = eta_f.segment(0, d0);
            loo   = lam_f.block(d0, d0, d1, d1);
            lono  = lam_f.block(d0, 0, d1, d0);
            lnoo  = lam_f.block(0, d0, d0, d1);
            lnono = lam_f.block(0, 0, d0, d0);
        }

        lnono.diagonal().array() += kJitter;

        Eigen::MatrixXd rhs(lnono.rows(), lnoo.cols() + 1);
        rhs.leftCols(lnoo.cols()) = lnoo;
        rhs.col(rhs.cols() - 1) = eno;

        Eigen::LLT<Eigen::MatrixXd> llt(lnono);
        if (llt.info() != Eigen::Success) {
            // 建议这里至少打印一下并中止，避免 silent NaN 传播
            throw std::runtime_error("LLT failed in Factor::computeMessages");
        }

        Eigen::MatrixXd X = llt.solve(rhs);
        Eigen::MatrixXd X_lam = X.leftCols(lnoo.cols());
        Eigen::VectorXd X_eta = X.col(X.cols() - 1);

        Eigen::MatrixXd msg_lam = loo - lono * X_lam;
        Eigen::VectorXd msg_eta = eo  - lono * X_eta;

        // Damping with OLD self message (Python uses old self.messages[target])
        if (target == 0) {
            msg_lam = (1.0 - eta_damping) * msg_lam + eta_damping * old_lam0;
            msg_eta = (1.0 - eta_damping) * msg_eta + eta_damping * old_eta0;
        } else {
            msg_lam = (1.0 - eta_damping) * msg_lam + eta_damping * old_lam1;
            msg_eta = (1.0 - eta_damping) * msg_eta + eta_damping * old_eta1;
        }

        new_lam[target] = msg_lam;
        new_eta[target] = msg_eta;
    }

    // ---- commit both at the end (synchronous) ----
    messages[0].setLam(new_lam[0]);  messages[0].setEta(new_eta[0]);
    messages[1].setLam(new_lam[1]);  messages[1].setEta(new_eta[1]);
}

} // namespace gbp
