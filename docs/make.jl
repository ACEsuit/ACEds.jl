# Instructions for building docs locally: 
#   - activate and resolve/up docs Project
#   - ] dev ..  to link to the *current* version of ACEpotentials
#   - julia --project=. make.jl  or   julia --project=docs docs/make.jl
#

using Documenter, ACEds

makedocs(sitename="My Documentation")