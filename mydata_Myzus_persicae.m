function [data, auxData, metaData, txtData, weights] = mydata_Myzus_persicae

%% set metaData
metaData.phylum     = 'Arthropoda'; 
metaData.class      = 'Insecta'; 
metaData.order      = 'Hemiptera'; 
metaData.family     = 'Aphididae';
metaData.species    = 'Myzus_persicae'; 
metaData.species_en = 'Green peach aphid'; 

metaData.ecoCode.climate = {'A', 'BS', 'Cs', 'Cwa', 'Cwb', 'Cfa', 'Cfb', 'Dfa', 'Dfb', 'Dfc'};
metaData.ecoCode.ecozone = {'TH',  'TPi', 'TPam','TPac', 'TA', 'TO'};
metaData.ecoCode.habitat = {'0iTh', '0iTf', '0iTi', '0iTs', '0iTg', '0iTa'};
metaData.ecoCode.embryo  = {'Tv', 'Th'};
metaData.ecoCode.migrate = {};
metaData.ecoCode.food    = {'biHl'};
metaData.ecoCode.gender  = {'D'};
metaData.ecoCode.reprod  = {'Apf'};

metaData.T_typical  = C2K(20); % K, body temp
metaData.data_0     = {'aj'; 'am'; 'Lb'; 'Lj'; 'Wwb'; 'Ww_t'; 'Ri'};
metaData.data_1     = {'T-tj'; 'T-am'; 'T-Ri'; 'T-Ww'};

metaData.COMPLETE = 2.5; % using criteria of LikaKear2011

metaData.author   = {'Josef Koch','Jana Gerhard','Christian Baden','Andre Gergs'};         
metaData.date_subm = [2024 11 25];
metaData.email    = {'koch@gaiac-eco.de'};        
metaData.address  = {'gaiac - Research Institute for Ecosystem Analysis and Assessment, 52072 Aachen, Germany'};  

metaData.curator     = {'Bas Kooijman'};
metaData.email_cur   = {'bas.kooijman@vu.nl'}; 
metaData.date_acc    = [2024 11 026];

%% set data
% zero-variate data

data.tj = 7.84;   units.tj = 'd';   label.tj = 'time since birth at adult emergence';   bibkey.tj = 'DaviRadc2007';
   temp.tj = C2K(20);   units.temp.tj = 'K';   label.temp.tj = 'temperature';
data.t1 = 2.42;    units.t1 = 'd';    label.t1 = 'development time instar 1';    bibkey.t1 = 'Fenj1945';   
   temp.t1 = C2K(18);    units.temp.t1 = 'K';    label.temp.t1 = 'temperature';
data.t2 = 2.42;    units.t2 = 'd';    label.t2 = 'development time instar 2';    bibkey.t2 = 'Fenj1945';   
   temp.t2 = C2K(18);    units.temp.t2 = 'K';    label.temp.t2 = 'temperature';
data.t3 = 2.46;    units.t3 = 'd';    label.t3 = 'development time instar 3';    bibkey.t3 = 'Fenj1945';   
   temp.t3 = C2K(18);    units.temp.t3 = 'K';    label.temp.t3 = 'temperature';
data.t4 = 2.58;    units.t4 = 'd';    label.t4 = 'development time instar 4';    bibkey.t4 = 'Fenj1945';   
   temp.t4 = C2K(18);    units.temp.t4 = 'K';    label.temp.t4 = 'temperature';
data.am = 5.4;    units.am = 'd';    label.am = 'life span of an imago';    bibkey.am = 'CutlRama2009';   
   temp.am = C2K(25);    units.temp.am = 'K';    label.temp.am = 'temperature';   

data.Lb = 0.0779;   units.Lb = 'cm';   label.Lb = 'body length at birth';   bibkey.Lb = 'Sylv1954';
data.L2 = 0.0834;   units.L2 = 'cm';   label.L2 = 'body length of instar 2';   bibkey.L2 = 'Sylv1954';
data.L3 = 0.1062;   units.L3 = 'cm';   label.L3 = 'body length of instar 3';   bibkey.L3 = 'Sylv1954';
data.L4 = 0.1466;   units.L4 = 'cm';   label.L4 = 'body length of instar 4';   bibkey.L4 = 'Sylv1954';
data.Lj = 0.1957;   units.Lj = 'cm';   label.Lj = 'body length at imago stage';   bibkey.Lj = 'Sylv1954';

data.Wwb = 0.00003326;   units.Wwb = 'g';   label.Wwb = 'wet weight at birth';   bibkey.Wwb = 'Dadd1967';
   comment.Wwb = 'alatiform data used assuming equal initial weights for both viviparous forms';
data.Ww9d = 0.0006629;   units.Ww9d = 'g';   label.Ww9d = 'wet weight after 9 days';   bibkey.Ww9d = 'CaoZhan2018';
   comment.Ww9d = '9 day old aphids at 22 degC, fed with young leaves of cabbage, approx. 47 adults and 3 nymphs';

data.Ri20 = 1.66;   units.Ri20 = '#/d';   label.Ri20 = 'maximum reproduction rate';   bibkey.Ri20 = 'DaviRadc2007';   
  temp.Ri20 = C2K(20);   units.temp.Ri20 = 'K';   label.temp.Ri20 = 'temperature';
  comment.Ri20 = 'recorded over 7 days; cultivar: Red La Soda';
data.Ri25 = 1.65;   units.Ri25 = '#/d';   label.Ri25 = 'maximum reproduction rate';   bibkey.Ri25 = 'CutlRama2009';   
  temp.Ri25 = C2K(25);   units.temp.Ri25 = 'K';   label.temp.Ri25 = 'temperature';
  comment.Ri25 = 'recorded over 2 days';

% uni-variate data

% temperature-time since birth at imago stage
data.Ttj_1 = [ ... temperature (C), time since birth at imago stage (days)
15    11.6
17.5  9.2
20    7.1
22.5  5.9
25    5.1
27.5  6.1
30    6.4];
units.Ttj_1 = {'C', 'd'};  label.Ttj_1 = {'temperature', 'time since birth at imago emergence', 'pepper'};  
bibkey.Ttj_1 = 'SataKers2008';
% 
data.Ttj_2 = [ ... temperature (C), time since birth at imago stage (days)
10    20.88
16    11.41
21    8.33
24    6.49
28    7.25];
units.Ttj_2 = {'C', 'd'};  label.Ttj_2 = {'temperature', 'time since birth at imago emergence', 'spinach'};  
bibkey.Ttj_2 = 'Weed1927';
% 
data.Ttj_3 = [ ... temperature (C), time since birth at imago stage (days)
% 6.2   43 % from 15 individuals, only one reached the imago stage after 43 days
9.9   24.5
13.5  17
15.2  15
18    10
21.2  7.5
25    8
28.3  11];
units.Ttj_3 = {'C', 'd'};  label.Ttj_3 = {'temperature', 'time since birth at imago emergence', 'beet'};  
bibkey.Ttj_3 = 'Fenj1945';

% temperature-time from birth to instar 1/2 transition 
data.Tt1 = [ ... temperature (C), time from birth to instar 1/2 transition (d)
10	5.25
16	3.14
21	1.98
24	1.63
28	1.50];
units.Tt1 = {'C', 'd'};  label.Tt1 = {'temperature', 'development time per instar', 'from b to i 1/2'};  
bibkey.Tt1 = 'Weed1927'; comment.Tt1 = 'food source: spinach';
% temperature-time from instar 1/2 to 2/3 transition
data.Tt2 = [ ... temperature (C), time from instar 2 to instar 3 (days)
10	5.00
16	2.82
21	2.05
24	1.60
28	2.00];
units.Tt2 = {'C', 'd'};  label.Tt2 = {'temperature', 'development time per instar', ' from i 1/2 to i 2/3'};  
bibkey.Tt2 = 'Weed1927'; comment.Tt2 = 'food source: spinach';
% temperature-time from instar 2/3 to 3/4 transition
data.Tt3 = [ ... temperature (C), time from instar 3 to instar 4 (days)
10	5.25
16	2.50
21	2.01
24	1.42
28	1.50];
units.Tt3 = {'C', 'd'};  label.Tt3 = {'temperature', 'development time per instar', 'from i 2/3 to i 3/4'};  
bibkey.Tt3 = 'Weed1927'; comment.Tt3 = 'food source: spinach';
% temperature-time from instar 3/4 to 4/imago transition
data.Tt4 = [ ... temperature (C), time from instar 4 to imago (days)
10	5.38
16	2.92
21	2.29
24	1.85
28	2.25];
units.Tt4 = {'C', 'd'};  label.Tt4 = {'temperature', 'development time per instar', 'from i 3/4 to i 4/imago'};  
bibkey.Tt4 = 'Weed1927'; comment.Tt4 = 'food source: spinach';

% temperature-life span
data.Tam1 = [ ... temperature (C), average life span (days)
25.0	36.8
27.5	31.6
30.0	18.0
32.5	5.4];
units.Tam1 = {'C', 'd'};  label.Tam1 = {'temperature', 'average life span since birth', 'pepper'};  
bibkey.Tam1 = 'SataKers2008';
% 
data.Tam2 = [ ... temperature (C), average life span (days)
15	9.50
20	11.50
30	9.25
32	3.5];
units.Tam2 = {'C', 'd'};  label.Tam2 = {'temperature', 'average life span since birth', 'Chinese cabbage'};  
bibkey.Tam2 = 'DaviRadc2006';
% 
data.Tam3 = [ ... temperature (C), average life span (days)
10	52.91
16	40.60
21	30.66
24	26.00
28	17.83];
units.Tam3 = {'C', 'd'};  label.Tam3 = {'temperature', 'average life span since birth', 'spinach'};  
bibkey.Tam3 = 'Weed1927';
% temperature
data.Tam4 = [ ... temperature (C), average life span (days)
9.9	    69
15.2	61
18.0	52
25.2	21
28.3	16];
units.Tam4 = {'C', 'd'};  label.Tam4 = {'temperature', 'average life span since birth', 'beet'};  
bibkey.Tam4 = 'Fenj1945';

% temperature-preoviposition period
data.Tt0R1 = [ ... temperature (C), preoviposition period (days)
10	3.60
16	1.44
21	1.13
24	0.92
28	1.00];
units.Tt0R1 = {'C', 'd'};  label.Tt0R1 = {'temperature', 'preoviposition period', 'spinach'};  
bibkey.Tt0R1 = 'Weed1927';
% 
data.Tt0R2 = [ ... temperature (C), preoviposition period (days)
9.7 	4.5
15.0	2.0
18  	1.0
25.4	0.8
28.3	0.7];
units.Tt0R2 = {'C', 'd'};  label.Tt0R2 = {'temperature', 'preoviposition period', 'beet'};  
bibkey.Tt0R2 = 'Fenj1945';

% temperature-reproduction rate
data.TRi1 = [... % temperature (C), repro (#/d)
15.0	2.90
17.5	4.01
20.0	5.72
22.5	6.57
25.0	6.20
27.5	4.95];
units.TRi1  = {'C', '#/d'};  label.TRi1 = {'temperature', 'maximum reproduction rate', 'pepper'};  
bibkey.TRi1 = 'SataKers2008';
% 
data.TRi2 = [... % temperature (C), repro (#/d)
15	1.76
20	1.20
30	1.40];
units.TRi2  = {'C', '#/d'};  label.TRi2 = {'temperature', 'maximum reproduction rate', 'Chinese cabbage'};  
bibkey.TRi2 = 'DaviRadc2006';
% 
data.TRi3 = [... % temperature (degC), repro (#/d)
10	1.79
16	2.62
21	3.23
24	3.72
28	3.07];
units.TRi3  = {'C', '#/d'};  label.TRi3 = {'temperature', 'maximum reproduction rate', 'spinach'};  
bibkey.TRi3 = 'Weed1927';
% 
data.TRi4 = [... % temperature (C), repro (#/d)
9.9	    1.1
15.2	1.9
18.0	2.5
25.2	3.3
28.3	3];
units.TRi4  = {'C', '#/d'};  label.TRi4 = {'temperature', 'maximum reproduction rate', 'beet'};  
bibkey.TRi4 = 'Fenj1945';

% time-wet weight
data.tWw = [ ... % time since birth (d), gained wet weight (g)
1	0.000027
2	0.000044
3	0.0001
4	0.00018
5	0.000308
6	0.000426
7	0.000535
8	0.000628];
units.tWw   = {'d', 'g'};  label.tWw = {'time since birth', 'wet weight gained since birth'};  
temp.tWw    = C2K(20);  units.temp.tWw = 'K'; label.temp.tWw = 'temperature';
bibkey.tWw = 'Mitt1970';   comment.tWw = 'based on estimates of honeydew uptake';
  
%% set weights for all real data
weights = setweights(data, []);
weights.Ttj_2 = 0 * weights.Ttj_2; % data already represented in durational data for individual instars; avoiding double weighting of the same data; still included for plots
weights.Tam1 = 0 * weights.Tam1; % temperature-dependency in life span data contradicts development and reproduction data; requires additional stress module; excluded for now
weights.Tam2 = 0 * weights.Tam2; % temperature-dependency in life span data contradicts development and reproduction data; requires additional stress module; excluded for now
weights.Tam3 = 0 * weights.Tam3; % temperature-dependency in life span data contradicts development and reproduction data; requires additional stress module; excluded for now
weights.Tam4 = 0 * weights.Tam4; % temperature-dependency in life span data contradicts development and reproduction data; requires additional stress module; excluded for now

%% set pseudodata and respective weights
[data, units, label, weights] = addpseudodata(data, units, label, weights);

%% pack auxData and txtData for output
auxData.temp = temp;
txtData.units = units;
txtData.label = label;
txtData.bibkey = bibkey;
txtData.comment = comment;

%% Group plots
set1 = {'Ttj_1', 'Ttj_2', 'Ttj_3'}; subtitle1 = {'Data for pepper, spinach, beet'};
set2 = {'TRi1', 'TRi2', 'TRi3', 'TRi4'}; subtitle2 = {'Data for pepper, Chinese cabbage, spinach, beet'};
set3 = {'Tam1', 'Tam2', 'Tam3', 'Tam4'}; subtitle3 = {'Data for pepper, Chinese cabbage, spinach, beet'};
set4 = {'Tt0R1', 'Tt0R2'}; subtitle4 = {'Data for spinage, beet'};
set5 = {'Tt4', 'Tt3','Tt2', 'Tt1'}; subtitle5 = {'Data for instar 4, 3, 2, 1'};
metaData.grp.sets = {set1, set2, set3, set4, set5};
metaData.grp.subtitle = {subtitle1, subtitle2, subtitle3, subtitle4, subtitle5};

%% Discussion points
D1 = 'For now, this entry focuses only on apterous viviparous females of the summer cycle, i.e., the most common form used in laboratory studies';
D2 = 'Investment into reproduction starts at birth, thus the hex model was used instead of abp';
D3 = 'The pupa phase of the model type hex is skipped, so that the event j indicates the final moult to the imago stage instead; consequently, parameters kap_V and E_He remain unused';
D4 = 'Imagos are assumed to behave like in model type abp: growth stop, no kappa rule, continued feeding and reproduction';
D5 = 'The release rate of offspring is assumed to match the repro investment of the imago 1:1, so d_ER is net zero, and ERj remains as a buffer (see next discussion point)';
D6 = 'All repro investment before the imago stage serves as a time-buffer to have fully developed embryos ready by the end of the preoviposition period';
D7 = 'The preoviposition period is covered by a new parameter t_0R ("preoviposition period at reference temperature"), which should be corrected for temperature but which for now is independent of f';
D8 = 'An upper boundary was used to capture reduced development and reproduction around 30 C';
D9 = 'Temperature-dependent life-span data do not match the upper boundary assumption, indicating increased stress at higher temperatures instead; for now, all T-am wheights were set to zero';
D10 = 'Parameter kap was fixed at 0.8 to avoid convergence to 1; data on repro buffer at final moult would be needed to avoid this';
D11 = 'No maternal effects were assumed when calculating egg costs at different f-levels; otherwise differences in reproduction rates using different food types could not be explained well enough';
D12 = 'Large range of f-values needed to depict the temperature-dependent reproduction rates on various host plants; f = 1 used for most of the 0-var data where common hosts have been used; f > 1 were allowed for the other host plants/food types';
metaData.discussion = struct('D1',D1, 'D2',D2, 'D3',D3, 'D4',D4, 'D5',D5, 'D6',D6, 'D7',D7, 'D8',D8, 'D9',D9, 'D10',D10, 'D11',D11, 'D12',D12);

%% Facts
F1 = 'Autumn/winter cycle has sexual reproduction with egg-laying and is one-generation only';
metaData.bibkey.F1 = 'Capi2001'; 
F2 = 'Summer cycle can last for many generations and is clonal and viviparous';
metaData.bibkey.F2 = 'Capi2001'; 
F3 = 'Allocation to reproduction (and development of new embryos) starts early during larval development, likely already at birth';
metaData.bibkey.F3 = 'SearMitt1980';
F4 = 'Nymphs reach image in 4 molts with no pupa stage';
metaData.bibkey.F4 = 'Capi2001';
metaData.facts = struct('F1',F1, 'F2',F2, 'F3',F3, 'F4',F4);

%% Acknowledgment
metaData.acknowledgment = 'The creation of this entry was financially supported by Bayer AG';

%% Links
metaData.links.id_CoL = '45J3R'; % Cat of Life
metaData.links.id_ITIS = '200635'; % ITIS
metaData.links.id_EoL = '588681'; % Ency of Life
metaData.links.id_Wiki = 'Myzus_persicae'; % Wikipedia
metaData.links.id_ADW = 'Myzus_persicae'; % ADW
metaData.links.id_Taxo = '19508'; % Taxonomicon
metaData.links.id_WoRMS = '990733'; % WoRMS

%% References
bibkey = 'Wiki'; type = 'Misc'; bib = ...
'howpublished = {\url{http://en.wikipedia.org/wiki/Myzus_persicae}}';
metaData.biblist.(bibkey) = ['''@', type, '{', bibkey, ', ' bib, '}'';'];
%
bibkey = 'Kooy2010'; type = 'Book'; bib = [ ...  % used in setting of chemical parameters and pseudodata
'author = {Kooijman, S.A.L.M.}, ' ...
'year = {2010}, ' ...
'title  = {Dynamic Energy Budget theory for metabolic organisation}, ' ...
'publisher = {Cambridge Univ. Press, Cambridge}, ' ...
'pages = {Table 4.2 (page 150), 8.1 (page 300)}, ' ...
'howpublished = {\url{../../../bib/Kooy2010.html}}'];
metaData.biblist.(bibkey) = ['''@', type, '{', bibkey, ', ' bib, '}'';'];
%
bibkey = 'Capi2001'; type = 'techreport'; bib = [ ...
'author = {Capinera, J.L.}, ' ... 
'year = {2001}, ' ...
'title = {Green Peach Aphid, \emph{Myzus persicae} ({S}ulzer) ({I}nsecta: {H}emiptera: {A}phididae)}, ' ...
'institution = {Institute of Food and Agricultural Sciences, University of Florida}, ' ...
'howpublished = {\url{https://edis.ifas.ufl.edu/publication/IN379}}'];
metaData.biblist.(bibkey) = ['''@', type, '{', bibkey, ', ' bib, '}'';'];
%
bibkey = 'SearMitt1980'; type = 'Article'; bib = [ ...
'author = {Searle, J.B. and Mittler, T.E.}, ' ... 
'year = {1980}, ' ...
'title = {Embryogenesis and the production of males by apterous viviparae of the green peach aphid \emph{Myzus persicae} in relation to photoperiod}, ' ...
'journal = {Journal of Insect Physiology}, ' ...
'volume = {27}, ' ...
'number = {2}, ' ...
'pages = {145-153}'];
metaData.biblist.(bibkey) = ['''@', type, '{', bibkey, ', ' bib, '}'';'];
%
bibkey = 'DaviRadc2007'; type = 'Article'; bib = [ ...
'author = {Davis, J.A. and Radcliffe, E.B. and Ragsdale, D.W.}, ' ... 
'year = {2007}, ' ...
'title = {Resistance to green peach aphid, \emph{Myzus persicae} ({S}ulzer), and potato aphid, \emph{Macrosiphum euphorbiae} ({T}homas), in potato cultivars}, ' ...
'journal = {American Journal of Potato Research}, ' ...
'volume = {84}, ' ...
'pages = {259-269}'];
metaData.biblist.(bibkey) = ['''@', type, '{', bibkey, ', ' bib, '}'';'];
%
bibkey = 'Fenj1945'; type = 'article'; bib = [ ...
'author = {Fenjves, P.}, ' ... 
'year = {1945}, ' ...
'title = {Beitraege zur Kenntnis der Blattlaus \emph{Myzus (Myzodes) persicae} {S}ulz., Uebertraegerin der Blattrollkrankheit der Kartoffel}, ' ...
'journal = {Mitteilungen der Schweizerischen entomologischen Gesellschaft}, ' ...
'volume = {19}, ' ...
'number = {11}, ' ...
'howpublished = {\url{https://www.research-collection.ethz.ch/handle/20.500.11850/135483}}'];
metaData.biblist.(bibkey) = ['''@', type, '{', bibkey, ', ' bib, '}'';'];
%
bibkey = 'CutlRama2009'; type = 'Article'; bib = [ ...
'author = {Cutler, G.C. and Ramanaidu, K. and Astatkie, T. and Isman, M.B.}, ' ... 
'year = {2009}, ' ...
'title = {Green peach aphid, \emph{Myzus persicae} ({H}emiptera: {A}phididae), reproduction during exposure to sublethal concentrations of imidacloprid and azadirachtin}, ' ...
'journal = {Pest Management Science}, ' ...
'volume = {65}, ' ...
'pages = {205-209}'];
metaData.biblist.(bibkey) = ['''@', type, '{', bibkey, ', ' bib, '}'';'];
%
bibkey = 'Sylv1954'; type = 'Article'; bib = [ ...
'author = {Sylvester, E.S.}, ' ... 
'year = {1954}, ' ...
'title = {Insectary Life History and Apterous Instar Morphology of \emph{Myzus Persicae} ({S}ulzer) ({H}omoptera, {A}phidae)}, ' ...
'journal = {Annals of the Entomological Society of America}, ' ...
'volume = {47}, ' ...
'number = {3}, ' ...
'pages = {397-406}'];
metaData.biblist.(bibkey) = ['''@', type, '{', bibkey, ', ' bib, '}'';'];
%
bibkey = 'Dadd1967'; type = 'Article'; bib = [ ...
'author = {Dadd, R.H.}, ' ... 
'year = {1967}, ' ...
'title = {Improvement of synthetic diet for the aphid \emph{Myzus persicae} using plant juices, nucleic acids, or trace metals}, ' ...
'journal = {Journal of Insect Physiology}, ' ...
'volume = {13}, ' ...
'number = {5}, ' ...
'pages = {763-778}'];
metaData.biblist.(bibkey) = ['''@', type, '{', bibkey, ', ' bib, '}'';'];
%
bibkey = 'CaoZhan2018'; type = 'Article'; bib = [ ...
'author = {Cao, H.H. and Zhang, Z.F. and Wang, X.F. and Liu, T.X.}, ' ...
'year = {2018}, ' ...
'title = {Nutrition versus defense: Why \emph{Myzus persicae} (green peach aphid) prefers and performs better on young leaves of cabbage}, ' ...
'journal = {PLoS One}, ' ...
'volume = {13}, ' ...
'number = {4}, ' ...
'pages = {e0196219}'];
metaData.biblist.(bibkey) = ['''@', type, '{', bibkey, ', ' bib, '}'';'];
%
bibkey = 'SataKers2008'; type = 'Article'; bib = [ ...
'author = {Satar, S. and Kersting, U. and Uygun, N.}, ' ...
'year = {2008}, ' ...
'title = {Effect of temperature on population parameters of \emph{Aphis gossypii} {G}lover and \emph{Myzus persicae} ({S}ulzer) ({H}omoptera: {A}phididae) on pepper}, ' ...
'journal = {Journal of Plant Diseases and Protection}, ' ...
'volume = {115}, ' ...
'number = {2}, ' ...
'pages = {69-74}'];
metaData.biblist.(bibkey) = ['''@', type, '{', bibkey, ', ' bib, '}'';'];
%
bibkey = 'Weed1927'; type = 'Article'; bib = [ ...
'author = {Weed, A.}, ' ...
'year = {1927}, ' ...
'title = {Metamorphosis and Reproduction in Apterous Forms of \emph{Myzus Persicae} {S}ulzer as Influenced By Temperature and Humidity}, ' ...
'journal = {Journal of Economic Entomology}, ' ...
'volume = {20}, ' ...
'number = {1}, ' ...
'pages = {150-157}'];
metaData.biblist.(bibkey) = ['''@', type, '{', bibkey, ', ' bib, '}'';'];
%
bibkey = 'DaviRadc2006'; type = 'Article'; bib = [ ...
'author = {Davis, J.A. and Radcliffe, E.B. and Ragsdale, D.W.}, ' ...
'year = {2006}, ' ...
'title = {Effects of High and Fluctuating Temperatures on \emph{Myzus persicae} ({H}emiptera: {A}phididae)}, ' ...
'journal = {Environmental Entomology}, ' ...
'volume = {35}, ' ...
'number = {6}, ' ...
'pages = {1461-1468}'];
metaData.biblist.(bibkey) = ['''@', type, '{', bibkey, ', ' bib, '}'';'];
%
bibkey = 'Mitt1970'; type = 'Article'; bib = [ ...
'author = {Mittler, T.E.}, ' ...
'year = {1970}, ' ...
'title = {Uptake Rates of Plant Sap and Synthetic Diet by the Aphid \emph{Myzus persicae}}, ' ...
'journal = {Annals of the Entomological Society of America}, ' ...
'volume = {63}, ' ...
'number = {6}, ' ...
'pages = {1701-1705}'];
metaData.biblist.(bibkey) = ['''@', type, '{', bibkey, ', ' bib, '}'';'];
%

