# Dataset Preprocessing README


## 🔥 June 2, 2025 Update for TUAB and TUEV

The previous preprocessing code for the **TUAB** and **TUEV** datasets was inherited from the [BIOT](https://github.com/ycq091044/BIOT) and [LaBraM](https://github.com/935963004/LaBraM) repositories. These original implementations included **random elements** in the data splitting process. Even with fixed random seeds, different hardware environments could lead to **inconsistent Train/Val/Test splits**. This issue has been carried forward into **CBraMod**.

In the performance comparison presented in the **CBraMod paper**, I directly cited the results reported in the **BIOT** and **LaBraM** papers without having access to their exact dataset splits. Therefore, I cannot guarantee that the comparisons were made using the **same dataset partitions**. As a result, the evaluation may **not be entirely fair**.

Moreover, others may also be unable to reproduce a **fair comparison** with **CBraMod** on **TUAB** and **TUEV** under the same conditions.

To fully address this issue, I have updated the preprocessing code for **TUAB** and **TUEV** to ensure **fixed, deterministic dataset splits**. If you are conducting experiments on these two datasets, please use the **latest version of the preprocessing code** to generate the splits.

For accurate and fair comparisons, it is **strongly recommended** to re-implement existing methods such as **BIOT**, **LaBraM**, and **CBraMod** **on the same fixed splits**.

> ⚠️ **Please note**: The **TUAB version** used in our experiments is **3.0.1**, and the **TUEV version** is **2.0.0**. Updates to the datasets may result in changes to the total number of samples.  
>  
> 📌 If you are using **different versions** of these datasets, **do not refer to our sample counts**. Instead, **reproduce the results directly on your own data splits**.
>
> ✅ We also provide **dataset splits** for **TUEV v2.0.1**. Please refer to the sample counts below for details. 

### 📊 Current Sample Counts (Updated Preprocessing)

#### **TUAB (v3.0.1):**
- **Train:** 297,103  
- **Validation:** 75,407  
- **Test:** 36,945  
- **Total:** 409,455


#### **TUEV (v2.0.0):**
- **Train:** 67,436  
- **Validation:** 15,634  
- **Test:** 29,421  
- **Total:** 112,491  


#### **TUEV (v2.0.1):**
- **Train:** 68,445  
- **Validation:** 15,487  
- **Test:** 29,421  
- **Total:** 113,353  