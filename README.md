# Mopo Case Studies

This repo is the complete workflow for the Mopo European Case Study, without the data.
The goal is to improve reproducibility and collaboration.

# First Time Set Up Instructions

ROUGH: First you need to install Julia with Juliaup. Juliaup is necessary for handling multiple versions of Julia and currently SpineOpt does not work with the most recent version of julia.

TODO: Also Python installation?

1. Clone this repository
1. Download the data zipped folder from Zenodo > Right click > Extract All > Choose the Raw-Data folder

1. EITHER open the project in VSCode and open a powershell terminal 

    OR open a regular powershell terminal and cd into the project folder: `cd [path to folder]`

1. Create & activate a new python environment:
    ```
    py -3.13 -m venv .venv
    .\venv\Scripts\Activate.ps1
    ```
1. Install python dependencies: 

    `python -m pip install -r requirements.txt`

1. Check that you have julia version 1.11:     
    
    `juliaup status` 
    - If 1.11 is not listed: `juliaup add 1.11.9`

1. Set the julia version for the project:

    `juliaup override set 1.11`

1. Set up julia dependencies in the project:

    `julia --project=. -e "using Pkg; Pkg.instantiate()"`

1. Run spinetoolbox: `spinetoolbox`

1. Open the project: *File > Open Project > Mopo-Case-Studies*

1. Double-click on each Julia tool and set the Project to the project folder 

    (This makes sure it sees the correct julia environment and packages)

1. Double-click on each intermediate datastore (pink icons) > *New SpineDB > Okay* 
    
    (This will create sqlite files in the default folders SpineToolbox chooses.)

TODO: Set to Consumer

TODO: Make sure tools use python env created above

# TODO Working Start-up

Once you've completed the first-time setup, this is how you can start-up when returning to work on the project.


# TODO Running the Workflow

- Launch spinetoolbox
- Tooling order
- Avoiding rerunning from raw
- Scenario filters
- Config files
