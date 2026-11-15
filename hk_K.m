function q = hk_K(u, G, dt)
%HK_K  Linear operator K of the Chambolle-Pock splitting for HK-dEIT.
%
%   q = HK_K(u, G, dt)
%
%   Primal u (struct):
%     u.rho  (N  x T+1)  nonneg. air content on graph nodes
%     u.m    (E  x T)    edge fluxes at half time steps
%     u.eta  (Ne x T+1)  non-transport residual (untouched by K)
%
%   Dual q = K u (struct):
%     q.th   (E x T)    theta_{k,e} = ( rb_{k,i} + rb_{k,j} ) / 2
%     q.m    (E x T)    identity copy of m
%     q.ze   (N x T)    zeta_k = (rho_{k+1}-rho_k)/dt + divG * m_k
%                       [- G.wv * c_k  if the ventilation mode is on: u.c
%                        (1 x T) is then a primal variable, see hk_core]
%     q.rb   (N x T)    rb_k = (rho_k + rho_{k+1})/2
%     q.pos  (N x T+1)  identity copy of rho (for the >=0 indicator)
%
%   Adjoint: hk_Kt.m (adjointness checked in test_hk_unit.m).
%
%   Author:  Hoang Nhut Huynh, National Chung Hsing University, Taichung, Taiwan
%   Project: HK-dEIT, Hellinger-Kantorovich regularized dynamic EIT for pendelluft
%            quantification (see README.md for the experiment map)
%   Core MATLAB (R2017b or later) and EIDORS v3.12 where noted; no toolboxes.

if isfield(G, 'absB'); aB = G.absB; else; aB = abs(G.B); end
rb    = 0.5 * (u.rho(:, 1:end-1) + u.rho(:, 2:end));   % N x T
q.th  = 0.5 * (aB * rb);                               % E x T
q.m   = u.m;                                           % E x T
q.ze  = diff(u.rho, 1, 2) / dt + G.divG * u.m;         % N x T
if isfield(G, 'wv') && ~isempty(G.wv)                   % v6 ventilation mode:
    q.ze = q.ze - G.wv * u.c;                           % zeta - w c(t)
end
q.rb  = rb;                                            % N x T
q.pos = u.rho;                                         % N x (T+1)
end
