function G = calc_dual_graph(fmdl, lung, opts)
%CALC_DUAL_GRAPH  Finite-volume dual graph of (a subset of) FEM elements. (v2)
%
%   G = CALC_DUAL_GRAPH(fmdl, lung)          one graph node per element
%   G = CALC_DUAL_GRAPH(fmdl, lung, opts)    optional voxel agglomeration
%
%   fmdl : EIDORS fwd_model (fields .nodes, .elems) or a struct with field
%          .fwd_model. Pure MATLAB - no EIDORS calls.
%   lung : element indices / logical mask of the graph region ([] = all)
%   opts : (optional)
%     .target_N  agglomerate the lung elements into ~target_N voxel
%                clusters (cube side tuned automatically). Graph nodes are
%                then clusters: vol = sum of element volumes, ell = total
%                shared face measure between two clusters, d = distance of
%                volume-weighted centroids. B*1 = 0 and the volume budget
%                are preserved exactly. Near-isotropic voxel clusters also
%                mitigate the mesh-isotropy issue of discrete transport
%                (Gladbach-Kopfer-Maas).
%     .groups    Ne x 1 labels (e.g. 1 = left lung, 2 = right lung); clusters
%                never mix labels.
%
%   Output struct G:
%     .node_idx  (Nl x 1) global indices of the lung elements
%     .cluster   (Nl x 1) graph node of each lung element (1:N if no
%                agglomeration)
%     .B (E x N) signed incidence (+1 at i, -1 at j);  .absB = |B|
%     .ell .d .dia (E x 1)  face measure, centroid distance, |diamond|
%     .vol (N x 1)          node volumes;  .cent (N x dim) node centroids
%     .divG (N x E) = diag(1./vol) B' diag(ell);  .D (E x N) = diag(sqrt(ell./d)) B
%     .EL (Ne x N)          node -> element map (value of node copied to
%                           each of its elements)
%     .N .E .Ne .edges (E x 2) .h_cluster (cube side, NaN if none)
%
%   Identities (test_hk_unit T2, T9):  B*1 = 0,  vol'*divG = 0.
%
%   Author:  Hoang Nhut Huynh, National Chung Hsing University, Taichung, Taiwan
%   Project: HK-dEIT, Hellinger-Kantorovich regularized dynamic EIT for pendelluft
%            quantification (see README.md for the experiment map)
%   Core MATLAB (R2017b or later) and EIDORS v3.12 where noted; no toolboxes.

if nargin < 3 || isempty(opts); opts = struct(); end
if isfield(fmdl, 'fwd_model'); fmdl = fmdl.fwd_model; end
nodes = fmdl.nodes;
elems = double(fmdl.elems);
[Ne, nd1] = size(elems);
dim = size(nodes, 2);
if nd1 ~= dim + 1
    error('calc_dual_graph: expecting simplex elements (dim+1 nodes each)');
end

% ---- lung selection -------------------------------------------------
if nargin < 2 || isempty(lung)
    node_idx = (1:Ne)';
elseif islogical(lung)
    node_idx = find(lung(:));
else
    node_idx = unique(lung(:));
end
Nl  = numel(node_idx);
g2l = zeros(Ne, 1);
g2l(node_idx) = 1:Nl;

% ---- element volumes and centroids (vectorized) --------------------
P1 = nodes(elems(:,1), :);
cent_all = zeros(Ne, dim);
for a = 1:nd1; cent_all = cent_all + nodes(elems(:,a), :); end
cent_all = cent_all / nd1;
if dim == 2
    v1 = nodes(elems(:,2),:) - P1;  v2 = nodes(elems(:,3),:) - P1;
    vol_all = abs(v1(:,1).*v2(:,2) - v1(:,2).*v2(:,1)) / 2;
else
    v1 = nodes(elems(:,2),:) - P1;  v2 = nodes(elems(:,3),:) - P1;
    v3 = nodes(elems(:,4),:) - P1;
    vol_all = abs(sum(v1 .* cross(v2, v3, 2), 2)) / 6;
end

% ---- face-based adjacency ------------------------------------------
C  = nchoosek(1:nd1, dim);
nf = size(C, 1);
F  = zeros(Ne*nf, dim);
owner = zeros(Ne*nf, 1);
for f = 1:nf
    rows = (f-1)*Ne + (1:Ne);
    F(rows, :)  = elems(:, C(f,:));
    owner(rows) = (1:Ne)';
end
[~, ~, ic] = unique(sort(F, 2), 'rows');
[ics, p] = sort(ic);
ow  = owner(p);
dup = ics(1:end-1) == ics(2:end);
ei  = ow([dup; false]);
ej  = ow([false; dup]);
fid = p([dup; false]);

keep = g2l(ei) > 0 & g2l(ej) > 0;
ei = ei(keep);  ej = ej(keep);  fid = fid(keep);
li = g2l(ei);   lj = g2l(ej);

% ---- face measure (vectorized) -------------------------------------
Pa = nodes(F(fid,1), :);  Pb = nodes(F(fid,2), :);
if dim == 2
    ell = sqrt(sum((Pb - Pa).^2, 2));
else
    Pc  = nodes(F(fid,3), :);
    ell = 0.5 * sqrt(sum(cross(Pb - Pa, Pc - Pa, 2).^2, 2));
end

vol_l  = vol_all(node_idx);
cent_l = cent_all(node_idx, :);

% ---- optional voxel agglomeration ----------------------------------
h_cluster = NaN;
if isfield(opts, 'target_N') && ~isempty(opts.target_N) && opts.target_N < Nl
    tN = opts.target_N;
    if isfield(opts, 'groups') && ~isempty(opts.groups)
        grp = opts.groups(:);  grp = grp(node_idx);
    else
        grp = ones(Nl, 1);
    end
    lo = min(cent_l, [], 1);
    hc = (sum(vol_l) / tN)^(1/dim);
    for rep = 1:10
        key = [grp, floor((cent_l - lo) / hc)];
        [~, ~, cid] = unique(key, 'rows');
        Nc = max(cid);
        if abs(Nc - tN) <= 0.05*tN; break; end
        hc = hc * (Nc / tN)^(1/dim);
    end
    h_cluster = hc;
    cvol  = accumarray(cid, vol_l);
    ccent = zeros(Nc, dim);
    for k = 1:dim
        ccent(:,k) = accumarray(cid, vol_l .* cent_l(:,k)) ./ cvol;
    end
    ci = cid(li);  cj = cid(lj);
    inter = ci ~= cj;
    pr = sort([ci(inter) cj(inter)], 2);
    [pairs, ~, pid] = unique(pr, 'rows');
    ell = accumarray(pid, ell(inter));
    li  = pairs(:,1);  lj = pairs(:,2);
    N   = Nc;  vol = cvol;  cent = ccent;  cluster = cid;
else
    N = Nl;  vol = vol_l;  cent = cent_l;  cluster = (1:Nl)';
end
E = numel(li);
d = sqrt(sum((cent(li,:) - cent(lj,:)).^2, 2));

% ---- assemble -------------------------------------------------------
G.node_idx = node_idx;
G.cluster  = cluster;
G.N  = N;   G.E = E;   G.Ne = Ne;
G.edges = [li lj];
G.B    = sparse([(1:E)'; (1:E)'], [li; lj], [ones(E,1); -ones(E,1)], E, N);
G.absB = abs(G.B);
G.ell  = ell;
G.d    = d;
if dim == 2;  G.dia = ell .* d / 2;  else;  G.dia = ell .* d / 3;  end
G.vol  = vol;
G.cent = cent;
G.divG = spdiags(1./G.vol, 0, N, N) * G.B' * spdiags(ell, 0, E, E);
G.D    = spdiags(sqrt(ell ./ d), 0, E, E) * G.B;
G.EL   = sparse(node_idx, cluster, 1, Ne, N);
G.h_cluster = h_cluster;
end
