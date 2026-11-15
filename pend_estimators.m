function [B, lag_ms, names] = pend_estimators(X, vol, sel, kmin, kei, dt)
%PEND_ESTIMATORS  Image-based pendelluft estimators: the baselines that HK-dEIT
%   is compared against, all computed on the SAME images, breaths and region
%   split.
%
%   [B, lag_ms, names] = pend_estimators(X, vol, sel, kmin, kei, dt)
%   X     N x T gas images (gas increase > 0, e.g. -GN), nodes = elements or clusters
%   vol   N x 1 node volumes
%   sel   N x 1 logical region S (e.g. anterior); its complement is region N
%   kmin  (nb+1) end-expiration frames, kei (nb) end-inspiration frames
%   dt    frame interval (s)
%   B     nb x 5, each in % of that breath's tidal volume (vol' * X):
%     1 opp    : sum_insp [min(dS+, dN-) + min(dN+, dS-)] - simultaneous
%                opposite regional volume changes (Otis / Yoshida pendelluft)
%     2 quadGN : sum_insp |dS - a dG|, a = <dG,dS>/<dG,dG> over the record -
%                the HK quadrature readout computed directly on the images
%     3 FRIC   : sum_insp sum_i vol_i max(-dx_i, 0)   (pixel volume losses; as
%                'FRIC-vol' in E2b/E4)
%     4 regout : sum_insp [max(-dS,0) + max(-dN,0)]   (regional losses; as
%                'region-out' in E2b/E4)
%     5 Sang   : sum_i vol_i [(max - min of x_i over the breath) - (x_i(kei) - x_i(kmin))]
%                pixel amplitude with own vs global timing (Sang et al.,
%                Ann Transl Med 2020; their per-pixel map summed over pixels)
%   lag_ms  delay of region N behind region S at the breath fundamental
%           (> 0: N later), from the whole record
%   d* are frame-to-frame changes, insp = frames kmin(b) .. kei(b).
%
%   Author:  Hoang Nhut Huynh, National Chung Hsing University, Taichung, Taiwan
%   Project: HK-dEIT, Hellinger-Kantorovich regularized dynamic EIT for pendelluft
%            quantification (see README.md for the experiment map)
%   Core MATLAB (R2017b or later) and EIDORS v3.12 where noted; no toolboxes.

names = {'opp', 'quadGN', 'FRIC', 'regout', 'Sang'};
vol = vol(:);  sel = logical(sel(:));  kmin = kmin(:)';  kei = kei(:)';
nb = numel(kei);
VS = vol(sel)' * X(sel, :);  VN = vol(~sel)' * X(~sel, :);  VG = VS + VN;
dS = diff(VS);  dN = diff(VN);  dG = diff(VG);
a  = (dG * dS') / max(dG * dG', eps);
r  = dS - a * dG;
B  = nan(nb, 5);
for b = 1:nb
    st = kmin(b):kei(b)-1;
    TV = VG(kei(b)) - VG(kmin(b));
    if numel(st) < 2 || TV <= 0; continue; end
    d  = diff(X(:, kmin(b):kei(b)), 1, 2);
    xb = X(:, kmin(b):kmin(b+1));
    B(b,:) = 100 / TV * [sum(min(max(dS(st),0), max(-dN(st),0)) + min(max(dN(st),0), max(-dS(st),0))), ...
                         sum(abs(r(st))), ...
                         sum(sum(vol .* max(-d, 0))), ...
                         sum(max(-dS(st),0) + max(-dN(st),0)), ...
                         sum(vol .* ((max(xb,[],2) - min(xb,[],2)) - (X(:,kei(b)) - X(:,kmin(b)))))];
end
% lag at the breath fundamental (period = median breath length)
P  = median(diff(kmin));  t = 0:numel(VS)-1;  e = exp(-2i*pi*t/P);
zS = detrend(VS) * e.';  zN = detrend(VN) * e.';
lag_ms = -angle(zN / zS) * P / (2*pi) * dt * 1e3;
end
