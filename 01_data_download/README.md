# Downloading Data from GBIF

This folder contains the code to download data from GBIF. The user is **required to have a list of species names** and the code downloads media and meta data from GBIF for those corresponding species.


The following steps need to be executed in order:

## 1. Fetch unique keys
Step 1 involves fetching unique taxon keys for each species from GBIF Taxanomy backbone.

```bash
python 01_fetch_taxon_keys.py \
    --species_filepath species_lists/example2.csv \
    --column_name beetles_species_names \
    --output_filepath output_data/keys/example2_data.csv
```
The description of the arguments to the script:
* `--species_filepath`: The user's list of species names. Example species lists are provided in the `species_lists` folder. **Required**.
* `--column_name`: The column name in the above csv file containing the species' names. **Required**.
* `--output_filepath`: The output file path with csv as extension. **Required**.

## 2. Download data

Step 2 involves downloading data from GBIF. There are two scripts for this step:
- `02a_fetch_gbif_metamorphic_data.py`
- `02b_fetch_gbif_other_data.py`

If the user needs to download data for species - such as moths, butterflies or frogs - that have a metamorphic life cycle (eggs-larvae-pupa-adult), `02a_fetch_gbif_metamorphic_data.py` script ensures that images of only **adult** stage are downloaded. If this does not matter, for example the case of mammals, then `02b_fetch_gbif_other_data.py` should be used.

```bash
python 02a_fetch_gbif_metamorphic_data.py \
--write_directory output_data/gbif_data/ \
--species_key_filepath output_data/keys/example2_data.csv \
--max_images_per_species 500 \
--resume_session True
```

The description of the arguments to the script:

* `--write_directory`: Path to the folder to download the data. **Required**.
* `--species_key_filepath`: Path to the output csv file from `01-fetch_taxon_keys.py`. **Required**.
* `--max_images_per_species`: Maximum number of images to download for any species. Optional. **Default** is **500**.
* `--resume_session`: `True` or `False`, whether resuming a previously stopped downloading session. **Requried**.

It is quite possible to have a list of hundreds or thousands of species and maybe downloading half-a-million images. The downloading process is not too fast and can take days to complete in such cases. The script does not require to be executed in one continuous session and the data can be fetched in multiple downloading parts. If the user is resuming a previous downloading session, `True` should be passed to the `--resume_session` argument and `False` for downloading from scratch.

