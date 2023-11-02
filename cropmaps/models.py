import logging
import os
from typing import List, Union
import numpy as np
import rasterio
import pandas as pd
import gc
import math

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.model_selection import GridSearchCV

from cropmaps import logger
logging = logger.setup(name = __name__)

import warnings
warnings.filterwarnings("ignore")

def LandCover_Masking(landcover_mask:str, predicted:str, results_to:str, fname:str = "Predictions_Crops.tif"):
    """Masks classification results with WorldCover product to keep only areas with crops.

    Args:
        landcover_mask (str): Path to land cover map (returned from sts.sentimeseries.LandCover)
        predicted (str): Path to predicted image
        results_to (str): Store the results
        fname (str, optional): New image name. Defaults to "Predictions_Crops.tif".

    Returns:
        str: Path of the new image
    """
    from cropmaps.sentinels import sentinel2
    worldcover, _ = sentinel2.reproj_match(landcover_mask, predicted)
    worldcover = worldcover[0, :, :]
    with rasterio.open(predicted) as src:
        metadata = src.meta
        array = src.read(1)

    array[(worldcover != 10) & (worldcover != 20) & (worldcover != 30) & (worldcover != 40) & (worldcover != 60)]= 0
    
    path = os.path.join(results_to, fname)
    with rasterio.open(path, "w", **metadata) as dst:
        dst.write(array, 1)
    
    return path

def save_model(model, spath:str, save_as:str = "model.save")->str:
    """Save model to disk.

    Args:
        model (sklearn.object): Model to be saved
        spath (str): Path to save the model
        save_as (str, optional): Name of the file with .save extension. Defaults to "model.save"

    Returns:
        str: Path of the saved file
    """
    import pickle
    fpath = os.path.join(spath, save_as)
    pickle.dump(model, open(fpath, 'wb')) # save the model to disk

    return fpath

def load_model(fname:str)->Union[RandomForestClassifier, SVC]:
    """Loads a saved model to memory.

    Args:
        fname (str): Path to model path

    Returns:
        Union[RandomForestClassifier, SVC]: Restored model
    """
    
    import pickle
    model = pickle.load(open(fname, 'rb'))

    return model

def fill_confMatrix(ct:pd.DataFrame, labels:List)->pd.DataFrame:
    """Fill a confusion matrix with data.

    Args:
        ct (pd.DataFrame): Cross-tabulate object
        labels (List): Labels of the categories

    Returns:
        pd.DataFrame: Filled confusion matrix
    """
    rowColIdx = list(labels) + ['All']
    # Usefull when rows are missing
    notAlabel = [val for val in list(ct.index) if val not in rowColIdx]
    if len(ct.index) < len(rowColIdx) or len(notAlabel)>0:
        for label in labels:
            if label not in ct.index:
                ct.loc[label] = [0] * ct.shape[1]

        ct = ct.reindex(index=rowColIdx)

    # Usefull when columns are missing
    notAlabel = [val for val in list(ct.columns) if val not in rowColIdx]
    if len(ct.columns) < len(rowColIdx) or len(notAlabel)>0:
        for label in labels:
            if label not in ct.columns:
                ct[label] = [0] * ct.shape[0]

        ct = ct.reindex(columns=rowColIdx)

    # Add producer accuracy
    pa = [round(ct.loc[rvcd][rvcd] / ct.loc[rvcd]['All'] *100, 2) for rvcd in labels]
    pa = [0 if math.isnan(x) else x for x in pa]
    pa.append(round(np.nanmean(pa), 2))   # total PA
    ct['PA'] = pa

    # Add User acc
    ua = [round(ct.loc[rvcd][rvcd] / ct.loc['All'][rvcd] * 100, 2) for rvcd in labels]
    ua = [0 if math.isnan(x) else x for x in ua]
    ua.append(round(np.nanmean(ua), 2))   # total UA

    # Overall acc
    tp = [ct.loc[rvcd][rvcd] for rvcd in labels]
    oa = np.round( np.nansum(tp) / ct.loc['All']['All'] * 100, 2)    # OA
    ua.append(f"OA {oa}")
    ct.loc['UA'] = ua

    # add f1 column
    f1 = [round(
        (2*ct.loc[rvcd]['PA']*ct.loc['UA'][rvcd]) / (ct.loc[rvcd]['PA']+ct.loc['UA'][rvcd]),
        2) for rvcd in labels]
    f1 = [0 if math.isnan(x) else x for x in f1]
    f1.append(f"avg_f1 {round(np.nanmean(f1), 2)}") # total f1

    # test another calculation of f1
    f1_from_avgPAUA = round((2*ct.loc['All']['PA']*ct.loc['UA']['All']) / (ct.loc['All']['PA'] + ct.loc['UA']['All']), 2)
    f1.append(f"avgPUA_f1 {f1_from_avgPAUA}")
    ct['f1'] = f1

    return ct

def importance(band_desc:list, feature_importance:np.ndarray, store_to:str = None):
    """Get the importance of the data per band and per date.

    Args:
        band_desc (list): List of the bands
        feature_importance (np.ndarray): Feature importance from sklearn model.features_importance_
        store_to (str, optional): Folder to save results. If None then the importance is printed as information from the logger. Defaults to None
    """
    
    BANDS = np.unique([b.split('_')[2] for b in band_desc])
    DATES = np.unique([b.split('_')[1] for b in band_desc])
    bands_importance = pd.DataFrame(columns = BANDS)
    dates_importance = pd.DataFrame(columns=DATES)
    for b, imp in zip(band_desc, feature_importance):
        band = b.split('_')[2]
        date = b.split('_')[1]
        bands_importance = bands_importance._append([{band:imp}], ignore_index=True)
        dates_importance = dates_importance._append([{date:imp}], ignore_index=True)
    
    bands_importance = bands_importance.apply(lambda x: pd.Series(x.dropna().values))
    dates_importance = dates_importance.apply(lambda x: pd.Series(x.dropna().values))
    bands_importance.loc['Total'] = bands_importance.mean()
    dates_importance.loc['Total'] = dates_importance.mean()
    if store_to is None:
        logging.info("Bands Importance:")
        logging.info(bands_importance.loc['Total'])
        logging.info("-----------------")
        logging.info("Dates Importance:")
        logging.info(dates_importance.loc['Total'])
        logging.info("-----------------")
    else:
        bands_importance.loc["Total"].to_csv(os.path.join(store_to, "bands_importance.csv"), index = True)
        dates_importance.loc["Total"].to_csv(os.path.join(store_to, "dates_importance.csv"), index = True)

def random_forest_train(cube_path:str, gt_fpath:str, results_to:str, test_size:float = 0.33, n_jobs = -1, gridsearch:bool = False, parameters:dict = {"n_estimators": 200})->RandomForestClassifier:
    """Train a model with Random Forest classifier.

    Args:
        cube_path (str): Path to datacube
        gt_fpath (str): Path to ground truth data (As a raster. Use cropmaps.prepare_vector.burn to transform vector data to raster.)
        results_to (str): Path to store results
        test_size (float, optional): Test sample size. Defaults to 0.33
        gridsearch (bool, optional): Hyperparameter Tuning using GridSearchCV. Check here: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html#sklearn.model_selection.GridSearchCV. Defaults to False.
        parameters(dict, optional): SVM parameters. Check here: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html#sklearn.ensemble.RandomForestClassifier.
    Returns:
        RandomForestClassifier: The model
    """
    logging.info("Training RF model...This may take a while...")
    if not os.path.exists(results_to):
        os.makedirs(results_to)

    with rasterio.open(cube_path) as cb_src:
        cube_img = cb_src.read()
        band_desc = cb_src.descriptions

    with rasterio.open(gt_fpath) as gt_src:
        gt_img = gt_src.read(1)
        y_meta = gt_src.meta

    # Find how many non-zero entries we have -- i.e. how many training data samples?
    n_samples = (gt_img > 0).sum()

    # What are our classification labels?
    labels = np.unique(gt_img[gt_img > 0])
    X = cube_img[:, gt_img>0].astype("float32")
    y = gt_img[gt_img>0].astype("float32")
    X=X.T

    # Split train test datasets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size = test_size, random_state = 22227)

    logging.info(f'Cube matrix size: {cube_img.shape}')
    logging.info(f'GT matrix size: {gt_img.shape}')
    logging.info(f'Number of samples: {n_samples}, {round((n_samples / (gt_img.shape[0] * gt_img.shape[1])) * 100, 2)}% of total area in the image.')
    logging.info(f'The training data include {labels.size} classes.')
    logging.debug(f'X matrix size: {X.shape}')
    logging.debug(f'y array size: {y.shape}')
    
    gt_img = None
    cube_img = None
    gc.collect()
    
    if gridsearch == False:
        if "bootstap" in parameters:
            if parameters["bootstrap"]:
                oob_score = True
            else:
                oob_score = False
        else:
            oob_score = True
        # Initialize our model with 500 trees
        rf = RandomForestClassifier(**parameters, n_jobs = n_jobs, oob_score = oob_score)

        # Fit our model to training data
        rf = rf.fit(X_train.astype("float32"), y_train.astype("float32"))

        X_train = None
        gc.collect()

        # Calculate the importance of each band and each date
        importance(band_desc, rf.feature_importances_, store_to = results_to)

        # Setup a dataframe and predict on test dataset
        df = pd.DataFrame()
        df['truth'] = y_test
        df['predict'] = rf.predict(X_test)

        X_test = None
        gc.collect()
        if "bootstap" in parameters:
            if parameters["bootstrap"]:
                logging.info(f'OOB prediction of accuracy is: {round(rf.oob_score_ * 100, 2)}%')
        else:
            logging.info(f'OOB prediction of accuracy is: {round(rf.oob_score_ * 100, 2)}%')
    else:
        grid = GridSearchCV(RandomForestClassifier(), parameters, n_jobs = n_jobs)
        grid.fit(X_train, y_train)
        hyper_params = grid.best_params_
        if hyper_params["bootstrap"]:
            oob_score =  True
        else:
            oob_score = False
        rf = RandomForestClassifier(**hyper_params, n_jobs = n_jobs, oob_score = oob_score)
        # Fit our model to training data
        rf = rf.fit(X_train.astype("float32"), y_train.astype("float32"))
        
        X_train = None
        gc.collect()
        
        importance(band_desc, rf.feature_importances_, store_to = results_to)

        # Setup a dataframe and predict on test dataset
        df = pd.DataFrame()
        df['truth'] = y_test
        df['predict'] = rf.predict(X_test)
        
        X_test = None
        gc.collect()
        
        if hyper_params["bootstrap"]:
            logging.info(f'OOB prediction of accuracy is: {round(rf.oob_score_ * 100, 2)}%')
    
    # Cross-tabulate predictions
    cm = pd.crosstab(df['truth'], df['predict'], margins=True)
    cm = fill_confMatrix(cm, labels)
    cm.to_csv(os.path.join(results_to, "Confusion_Matrix.csv"))
    
    logging.info("Done.")

    return rf

def random_forest_predict(cube_path:str, model:RandomForestClassifier, results_to:str, fname:str = "Predictions.tif")->str:
    """Use to make predictions using a Random Forest model.

    Args:
        cube_path (str): Path to datacube
        model (RandomForestClassifier): The model
        results_to (str): Path to save results
        fname (str, optional): Name of the output predicted image. Defaults to "Predicts.tif"

    Returns:
        str: Path of the predicted image
    """
    logging.info("Predicting values with RF model...This may take a while...")

    if not os.path.exists(results_to):
        os.makedirs(results_to)
    
    # TODO: This is bad, must come up with a better solution
    with rasterio.open(cube_path) as cb_src:
        cube_img = cb_src.read()
        mask = np.any(cube_img != -9999, axis = 0)
        metadata = cb_src.meta.copy()    
        metadata.update(dtype = np.uint8, nodata = 0, count = 1)
        final_shape = (cube_img.shape[1], cube_img.shape[2])

    # Bands as last dimension
    cube_img = cube_img.transpose((1,2,0))
    # Take our full image and reshape into long 2d array (nrow * ncol, nband) for classification
    new_shape = (cube_img.shape[0] * cube_img.shape[1], cube_img.shape[2])

    cube_vector = cube_img.reshape(new_shape)
    logging.debug(f'Reshaped from {cube_img.shape} to {cube_vector.shape}')
    cube_img = None
    gc.collect()
    
    try:
        y_pred = model.predict(cube_vector)

        cube_vector = None
        model = None
        gc.collect()
        # Reshape our classification map
        y_pred = y_pred.reshape(final_shape)
        y_pred[mask == 0] = 0
        ypred_path= os.path.join(results_to, fname)
        with rasterio.open(ypred_path, "w", **metadata) as dst:
            dst.write(y_pred, 1)
        
        logging.info("Done.")
    
    except np.core._exceptions._ArrayMemoryError: # In case cube does not fit the memory
        raise MemoryError("Array cannot fit in memory. Use patch predictions (cropmaps.models.random_forest_predict_patches()) instead!")
    
    return ypred_path
    
def random_forest_predict_patches(cube_path: str, model: RandomForestClassifier, results_to: str, 
                          patch_size: tuple = (512, 512), fname: str = "Predictions.tif") -> str:
    """
    Use to make predictions using a RandomForestClassifier model with image patches.

    Args:
        cube_path (str): Path to datacube
        model (RandomForestClassifier): The model
        results_to (str): Path to save results
        patch_size (tuple, optional): Size of the image patches. Defaults to (512, 512).
        fname (str, optional): Name of the output predicted image. Defaults to "Predictions.tif"

    Returns:
        str: Path of the predicted image
    """
    logging.info("Predicting values with RF model using patches...This may take a while...")

    if not os.path.exists(results_to):
        os.makedirs(results_to)

    # Open the cube image using rasterio
    with rasterio.open(cube_path) as cb_src:
        cube_img = cb_src.read()
        # Calculate the number of patches in each dimension
        num_patches_x = cube_img.shape[1] // patch_size[0]
        num_patches_y = cube_img.shape[2] // patch_size[1]
        # Initialize an empty array to store final predictions
        final_prediction = np.zeros((cube_img.shape[1], cube_img.shape[2]), dtype=np.uint8)

        # Iterate through patches
        for i in range(num_patches_x):
            for j in range(num_patches_y):
                # Extract patch from the cube image
                start_x, end_x = i * patch_size[0], (i + 1) * patch_size[0]
                start_y, end_y = j * patch_size[1], (j + 1) * patch_size[1]

                # Handle edges by adjusting patch boundaries
                if i == num_patches_x - 1:
                    end_x = cube_img.shape[1]
                if j == num_patches_y - 1:
                    end_y = cube_img.shape[2]

                # Extract patch from the cube image
                patch = cube_img[:, start_x:end_x, start_y:end_y]
                patch_shape = (patch.shape[1], patch.shape[2])
                patch_vector = patch.transpose(1, 2, 0)
                new_shape = (patch_vector.shape[0] * patch_vector.shape[1], patch_vector.shape[2])
                patch_vector = patch_vector.reshape(new_shape)
                patch_prediction = model.predict(patch_vector)
                patch_prediction = patch_prediction.reshape(patch_shape)

                # Update the final prediction array with the patch prediction
                final_prediction[start_x:end_x, start_y:end_y] = patch_prediction
        
        # Write the final prediction to the output file
        metadata = cb_src.meta.copy()
        metadata.update(dtype=np.uint8, nodata=0, count=1)
        final_pred_path = os.path.join(results_to, fname)
        mask = np.any(cube_img != -9999, axis = 0)
        final_prediction[mask == 0] = 0
        with rasterio.open(final_pred_path, "w", **metadata) as dst:
            dst.write(final_prediction, 1)

    logging.info("Prediction complete. Results saved at: %s", final_pred_path)
    return final_pred_path

def svm_train(cube_path:str, gt_fpath:str, results_to:str, test_size:float = 0.33,  gridsearch:bool = False, parameters:dict = {"C": 1.0})->str:
    """Train a model with Random Forest classifier.

    Args:
        cube_path (str): Path to datacube
        gt_fpath (str): Path to ground truth data (As a raster. Use cropmaps.prepare_vector.burn to transform vector data to raster.)
        results_to (str): Path to store results
        test_size (float, optional): Test sample size. Defaults to 0.33
        gridsearch (bool, optional): Hyperparameter Tuning using GridSearchCV. Check here: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html#sklearn.model_selection.GridSearchCV. Defaults to False.
        parameters(dict, optional): SVM parameters. Check here: https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html

    Returns:
        svm.SVC: The model
    """
    logging.info("Training SVM model...This may take a while...")

    if not os.path.exists(results_to):
        os.makedirs(results_to)
    
    with rasterio.open(cube_path) as cb_src:
        cube_img = cb_src.read()
        band_desc = cb_src.descriptions

    with rasterio.open(gt_fpath) as gt_src:
        gt_img = gt_src.read(1)
        y_meta = gt_src.meta

    # Find how many non-zero entries we have -- i.e. how many training data samples?
    n_samples = (gt_img > 0).sum()

    # What are our classification labels?
    labels = np.unique(gt_img[gt_img > 0])
    X = cube_img[:, gt_img>0]
    y = gt_img[gt_img>0]
    X=X.T

    # Split train test datasets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size = test_size, random_state = 22227)

    logging.info(f'Cube matrix size: {cube_img.shape}')
    logging.info(f'GT matrix size: {gt_img.shape}')
    logging.info(f'Number of samples: {n_samples}, {round((n_samples / (gt_img.shape[0] * gt_img.shape[1])) * 100, 2)}% of total area in the image.')
    logging.info(f'The training data include {labels.size} classes.')
    logging.debug(f'X matrix size: {X.shape}')
    logging.debug(f'y array size: {y.shape}')
    
    gt_img = None
    cube_img = None
    gc.collect()
    
    if gridsearch == False:
        clf = svm.SVC(**parameters)
        clf.fit(X_train, y_train)

        gc.collect()
        # Setup a dataframe and predict on test dataset
        df = pd.DataFrame()
        df['truth'] = y_test
        df['predict'] = clf.predict(X_test)
        X_test = None
        gc.collect()
    else:
        grid = GridSearchCV(svm.SVC(), parameters)
        grid.fit(X_train, y_train)
        hyper_params = grid.best_params_
        
        clf = svm.SVC(**hyper_params)
        # Fit our model to training data
        clf = clf.fit(X_train, y_train)
        
        X_train = None
        gc.collect()
        
        # Setup a dataframe and predict on test dataset
        df = pd.DataFrame()
        df['truth'] = y_test
        df['predict'] = clf.predict(X_test)
        
        X_test = None
        gc.collect()

    # Cross-tabulate predictions
    cm = pd.crosstab(df['truth'], df['predict'], margins=True)
    cm = fill_confMatrix(cm, labels)
    cm.to_csv(os.path.join(results_to, "Confusion_Matrix.csv"))
    
    logging.info("Done.")

    return clf

def svm_predict(cube_path:str, model:svm.SVC, results_to:str, fname:str = "Predictions.tif")->str:
    """Use to make predictions using a SVM model.

    Args:
        cube_path (str): Path to datacube
        model (RandomForestClassifier): The model
        results_to (str): Path to save results
        fname (str, optional): Name of the output predicted image. Defaults to "Predicts.tif"

    Returns:
        str: Path of the predicted image
    """
    logging.info("Predicting values with SVM model...This may take a while...")

    if not os.path.exists(results_to):
        os.makedirs(results_to)
    
    # TODO: This is bad, must come up with a better solution
    with rasterio.open(cube_path) as cb_src:
        cube_img = cb_src.read()
        mask = np.any(cube_img != -9999, axis = 0)
        metadata = cb_src.meta.copy()    
        metadata.update(dtype = np.uint8, nodata = 0, count = 1)
        final_shape = (cube_img.shape[1], cube_img.shape[2])

    # Bands as last dimension
    cube_img = cube_img.transpose((1,2,0))
    # Take our full image and reshape into long 2d array (nrow * ncol, nband) for classification
    new_shape = (cube_img.shape[0] * cube_img.shape[1], cube_img.shape[2])

    cube_vector = cube_img.reshape(new_shape)
    logging.debug(f'Reshaped from {cube_img.shape} to {cube_vector.shape}')

    cube_img = None
    gc.collect()

    # Now predict for each pixel
    y_pred = model.predict(cube_vector)

    cube_vector=None
    gc.collect()
    try:
        # Reshape our classification map
        y_pred = y_pred.reshape(final_shape)
        y_pred[mask == 0] = 0
        ypred_path= os.path.join(results_to, fname)
        with rasterio.open(ypred_path, "w", **metadata) as dst:
            dst.write(y_pred, 1)
        
        logging.info("Done.")

    except np.core._exceptions._ArrayMemoryError: # In case cube does not fit the memory
        raise MemoryError("Array cannot fit in memory. Use patch predictions (cropmaps.models.svm_predict_patches()) instead!")

    return ypred_path

def svm_predict_patches(cube_path: str, model: svm.SVC, results_to: str, 
                          patch_size: tuple = (512, 512), fname: str = "Predictions.tif") -> str:
    """
    Use to make predictions using a SVM model with image patches.

    Args:
        cube_path (str): Path to datacube
        model (svm.SVC): The model
        results_to (str): Path to save results
        patch_size (tuple, optional): Size of the image patches. Defaults to (512, 512)
        fname (str, optional): Name of the output predicted image. Defaults to "Predictions.tif"

    Returns:
        str: Path of the predicted image
    """
    logging.info("Predicting values with RF model using patches...This may take a while...")

    if not os.path.exists(results_to):
        os.makedirs(results_to)

    # Open the cube image using rasterio
    with rasterio.open(cube_path) as cb_src:
        cube_img = cb_src.read()
        # Calculate the number of patches in each dimension
        num_patches_x = cube_img.shape[1] // patch_size[0]
        num_patches_y = cube_img.shape[2] // patch_size[1]
        # Initialize an empty array to store final predictions
        final_prediction = np.zeros((cube_img.shape[1], cube_img.shape[2]), dtype=np.uint8)

        # Iterate through patches
        for i in range(num_patches_x):
            for j in range(num_patches_y):
                # Extract patch from the cube image
                start_x, end_x = i * patch_size[0], (i + 1) * patch_size[0]
                start_y, end_y = j * patch_size[1], (j + 1) * patch_size[1]

                # Handle edges by adjusting patch boundaries
                if i == num_patches_x - 1:
                    end_x = cube_img.shape[1]
                if j == num_patches_y - 1:
                    end_y = cube_img.shape[2]

                # Extract patch from the cube image
                patch = cube_img[:, start_x:end_x, start_y:end_y]
                patch_shape = (patch.shape[1], patch.shape[2])
                patch_vector = patch.transpose(1, 2, 0)
                new_shape = (patch_vector.shape[0] * patch_vector.shape[1], patch_vector.shape[2])
                patch_vector = patch_vector.reshape(new_shape)
                patch_prediction = model.predict(patch_vector)
                patch_prediction = patch_prediction.reshape(patch_shape)

                # Update the final prediction array with the patch prediction
                final_prediction[start_x:end_x, start_y:end_y] = patch_prediction
        
        # Write the final prediction to the output file
        metadata = cb_src.meta.copy()
        metadata.update(dtype=np.uint8, nodata=0, count=1)
        final_pred_path = os.path.join(results_to, fname)
        mask = np.any(cube_img != -9999, axis = 0)
        final_prediction[mask == 0] = 0
        with rasterio.open(final_pred_path, "w", **metadata) as dst:
            dst.write(final_prediction, 1)

    logging.info("Prediction complete. Results saved at: %s", final_pred_path)
    return final_pred_path