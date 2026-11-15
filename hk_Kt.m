function u = hk_Kt(q, G, dt)
%HK_KT  Adjoint of hk_K (see hk_K.m for the variable layout).
%
%   u = HK_KT(q, G, dt)
%
%   Author:  Hoang Nhut Huynh, National Chung Hsing University, Taichung, Taiwan
%   Project: HK-dEIT, Hellinger-Kantorovich regularized dynamic EIT for pendelluft
%            quantification (see README.md for the experiment map)
%   Core MATLAB (R2017b or later) and EIDORS v3.12 where noted; no toolboxes.

[~, T] = size(q.m);
T1 = T + 1;
N  = G.N;

% adjoint into the "temporal average" space rb:
%   th = 0.5*|B|*rb  and  rb itself
if isfield(G, 'absB'); aB = G.absB; else; aB = abs(G.B); end
w = 0.5 * (aB' * q.th) + q.rb;                         % N x T

% rb_k = (rho_k + rho_{k+1})/2  -> each rho frame collects halves
R = zeros(N, T1);
R(:, 1:T)  = R(:, 1:T)  + 0.5 * w;
R(:, 2:T1) = R(:, 2:T1) + 0.5 * w;

% zeta_k = (rho_{k+1} - rho_k)/dt + divG*m_k
R(:, 1:T)  = R(:, 1:T)  - q.ze / dt;
R(:, 2:T1) = R(:, 2:T1) + q.ze / dt;

% identity copy of rho
R = R + q.pos;

u.rho = R;
u.m   = q.m + G.divG' * q.ze;
if isfield(G, 'wv') && ~isempty(G.wv)
    u.c = -(G.wv' * q.ze);        % adjoint of c -> -w c   (1 x T)
else
    u.c = zeros(1, T);
end
u.eta = zeros(G.Ne, T1);          % K does not act on eta
end
