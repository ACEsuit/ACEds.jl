c_all1 = params(mb)
set_params!(mb,c_all1)
c_all2 = params(mb)
c_all1 == c_all2
c_all1 = params(mb, true)
set_params!(mb,c_all1)
c_all2 = params(mb, true)
c_all1 == c_all2
c_all1 = params(mb, :onsite, true)
set_params!(mb,:onsite,c_all1)
c_all2 = params(mb, :onsite, true)
c_all1 == c_all2