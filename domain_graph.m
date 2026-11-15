function [Gc, sfx, msk] = domain_graph(fm, grp, N_lung, MASK)
%DOMAIN_GRAPH  Cluster graph of the reconstruction domain shared by run_E2*
%   and make_figs_HK, so that every script builds the same graph.
%
%   [Gc, sfx, msk] = domain_graph(fm, grp, N_lung, MASK)
%   grp    Ne x 1: 1 left lung, 2 right lung, any other value = tissue
%          (clusters never mix groups)
%   N_lung clusters for the lung-only domain (800 in E2-E2h)
%   MASK   'lungs'  : library lungs, N_lung clusters        -> sfx ''
%          'thorax' : every element (E2i: the library lungs hold ~1/3 of
%                     the reconstructed TIV); clusters scaled with volume,
%                     capped at 1000 as in E3                 -> sfx '_thorax'
%   sfx    suffix for result files, so lung-mask results are never overwritten
%
%   Author:  Hoang Nhut Huynh, National Chung Hsing University, Taichung, Taiwan
%   Project: HK-dEIT, Hellinger-Kantorovich regularized dynamic EIT for pendelluft
%            quantification (see README.md for the experiment map)
%   Core MATLAB (R2017b or later) and EIDORS v3.12 where noted; no toolboxes.

isLung = grp == 1 | grp == 2;
switch MASK
    case 'lungs'
        msk = isLung;  Nc = N_lung;  sfx = '';
    case 'thorax'
        msk = true(size(grp));  vol = get_elem_volume(fm);
        Nc  = min(1000, round(N_lung * sum(vol) / sum(vol(isLung))));  sfx = '_thorax';
    otherwise
        error('domain_graph: MASK must be ''lungs'' or ''thorax''');
end
Gc = calc_dual_graph(fm, find(msk), struct('target_N', Nc, 'groups', grp));
end
