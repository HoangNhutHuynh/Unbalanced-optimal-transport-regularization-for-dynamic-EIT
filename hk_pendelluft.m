function [QG, PV, Gamma] = hk_pendelluft(G, m, sel, dt, insp)
%HK_PENDELLUFT  Interface flux and pendelluft volume from HK-dEIT fluxes.
%
%   [QG, PV, Gamma] = HK_PENDELLUFT(G, m, sel, dt, insp)
%
%   G    : dual graph (calc_dual_graph)
%   m    : E x T edge fluxes (img.hk.flux)
%   sel  : logical (N x 1) or index set of graph nodes defining region S
%          (e.g. dependent/posterior lung, or left lung)
%   dt   : frame interval [s]
%   insp : logical (T x 1) frames counted for PV (e.g. inspiration, or a
%          breath-hold window). Default: all frames.
%
%   Sign convention: with B(e,i)=+1, B(e,j)=-1, a flux m_e > 0 moves mass
%   from node i to node j (div at i is positive, so rho_i decreases).
%
%   QG(k) = net flux INTO region S at half-step k:
%           QG(k) = - sum_{e in Gamma} ell_e * s_e * m_{k,e},
%   where s_e = sum of B(e,:) over S (= +1 if only the + endpoint is in S,
%   -1 if only the - endpoint is in S, 0 if the edge does not cross Gamma).
%
%   PV = dt * sum_{k in insp} max(QG(k), 0)      (gas moved into S)
%
%   Gamma: logical E x 1 mask of interface edges.
%
%   Check: vol'*rho restricted change of S equals dt*(QG + sources in S).
%
%   Author:  Hoang Nhut Huynh, National Chung Hsing University, Taichung, Taiwan
%   Project: HK-dEIT, Hellinger-Kantorovich regularized dynamic EIT for pendelluft
%            quantification (see README.md for the experiment map)
%   Core MATLAB (R2017b or later) and EIDORS v3.12 where noted; no toolboxes.

if nargin < 5 || isempty(insp); insp = true(size(m,2),1); end
if ~islogical(sel)
    tmp = false(G.N,1); tmp(sel) = true; sel = tmp;
end

s_e   = G.B * double(sel);          % E x 1 in {-1, 0, +1}
Gamma = (abs(s_e) == 1);

QG = -( (G.ell(Gamma) .* s_e(Gamma))' * m(Gamma, :) )';   % T x 1
PV = dt * sum( max(QG(insp), 0) );
end
