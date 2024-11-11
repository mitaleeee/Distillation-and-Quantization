from fastapi import FastAPI  # type: ignore
from pydantic import BaseModel  # type: ignore
from transformers import BertTokenizer, BertModel, DistilBertModel, DistilBertTokenizer  # type: ignore
import torch  # type: ignore
import torch.nn as nn  # type: ignore
import os
import time
from functools import partial

if torch.cuda.is_available():
    device = torch.device(type="cuda", index=0)
else:
    device = torch.device(type="cpu", index=0)


TEACHER_PATH = "google-bert/bert-base-uncased"
STUDENT_PATH = "distilbert/distilbert-base-uncased"
TEACHER_HIDDEN_DIM = 768
STUDENT_HIDDEN_DIM = 768
MAX_SEQ_LEN = 512

BASE_DIR = "./models"

BERT = "BERT"
D_BERT = "D-BERT"
Q_BERT = "Q-BERT"
QD_BERT = "QD-BERT"


class TeacherModel_BERT_Large(nn.Module):
    def __init__(self):
        super(TeacherModel_BERT_Large, self).__init__()
        self.bert_model = BertModel.from_pretrained(TEACHER_PATH)
        self.out = nn.Linear(in_features=TEACHER_HIDDEN_DIM, out_features=2)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, ids, mask, token_type_ids):
        o = self.bert_model(
            ids, attention_mask=mask, token_type_ids=token_type_ids, return_dict=False
        )
        out = self.out(o[1])
        out = self.softmax(out)
        return out


class StudentModel_DistilBERT_Base(nn.Module):
    def __init__(self):
        super(StudentModel_DistilBERT_Base, self).__init__()
        self.bert_model = DistilBertModel.from_pretrained(STUDENT_PATH)
        self.out = nn.Linear(STUDENT_HIDDEN_DIM, 2)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, ids, mask):
        o = self.bert_model(ids, attention_mask=mask, return_dict=False)
        o = o[0][:, 0]
        out = self.out(o)
        out = self.softmax(out)
        return out


TRAINED_MODEL_PATHS = {
    BERT: os.path.join(BASE_DIR, "teacher_weights.pth"),
    D_BERT: os.path.join(BASE_DIR, "student_weights.pth"),
    Q_BERT: os.path.join(BASE_DIR, "quantized_teacher_weights.pth"),
    QD_BERT: os.path.join(BASE_DIR, "quantized_student_weights.pth"),
}

TOKENIZERS = {
    BERT: BertTokenizer.from_pretrained(
        TEACHER_PATH, clean_up_tokenization_spaces=True
    ),
    D_BERT: DistilBertTokenizer.from_pretrained(
        STUDENT_PATH, clean_up_tokenization_spaces=True
    ),
    Q_BERT: BertTokenizer.from_pretrained(
        TEACHER_PATH, clean_up_tokenization_spaces=True
    ),
    QD_BERT: DistilBertTokenizer.from_pretrained(
        STUDENT_PATH, clean_up_tokenization_spaces=True
    ),
}


def load_model(instance, path):
    instance.load_state_dict(state_dict=torch.load(path, weights_only=True))
    return instance


MODELS = {
    BERT: load_model(TeacherModel_BERT_Large(), TRAINED_MODEL_PATHS[BERT]),
    D_BERT: load_model(StudentModel_DistilBERT_Base(), TRAINED_MODEL_PATHS[D_BERT]),
    Q_BERT: load_model(
        torch.ao.quantization.quantize_dynamic(TeacherModel_BERT_Large()),
        TRAINED_MODEL_PATHS[Q_BERT],
    ),
    QD_BERT: load_model(
        torch.ao.quantization.quantize_dynamic(StudentModel_DistilBERT_Base()),
        TRAINED_MODEL_PATHS[QD_BERT],
    ),
}


def generalized_pipeline(
    text, working_model, working_tokenizer, tti_needed=False, device_=device
):
    inputs = working_tokenizer.encode_plus(
        text,
        None,
        padding="max_length",
        add_special_tokens=True,
        return_attention_mask=True,
        max_length=MAX_SEQ_LEN,
        truncation=True,
    )
    ids = torch.tensor(inputs["input_ids"], dtype=torch.long, device=device_)
    masks = torch.tensor(inputs["attention_mask"], dtype=torch.long, device=device_)
    if tti_needed:
        tti = torch.tensor(inputs["token_type_ids"], dtype=torch.long, device=device_)
        if ids.dim() == 1 or masks.dim() == 1 or tti.dim() == 1:
            ids = ids.unsqueeze(0)
            masks = masks.unsqueeze(0)
            tti = tti.unsqueeze(0)
        start = time.time()
        outputs = working_model(ids=ids, mask=masks, token_type_ids=tti)
        end = time.time()
    else:
        if ids.dim() == 1 or masks.dim() == 1:
            ids = ids.unsqueeze(0)
            masks = masks.unsqueeze(0)
        start = time.time()
        outputs = working_model(ids, masks)
        end = time.time()
    print("Time elapsed: ", (end - start) * 1000, "ms")
    return "AI Generated" if torch.argmax(outputs).item() == 1 else "Human"


student_pipeline = partial(
    generalized_pipeline,
    working_model=MODELS[D_BERT].to("cpu"),
    working_tokenizer=TOKENIZERS[D_BERT],
    tti_needed=False,
    device_="cpu",
)

teacher_pipeline = partial(
    generalized_pipeline,
    working_model=MODELS[BERT].to("cpu"),
    working_tokenizer=TOKENIZERS[BERT],
    tti_needed=True,
    device_="cpu",
)

q_student_pipeline = partial(
    generalized_pipeline,
    working_model=MODELS[QD_BERT],
    working_tokenizer=TOKENIZERS[QD_BERT],
    tti_needed=False,
    device_="cpu",
)

q_teacher_pipeline = partial(
    generalized_pipeline,
    working_model=MODELS[Q_BERT],
    working_tokenizer=TOKENIZERS[Q_BERT],
    tti_needed=True,
    device_="cpu",
)


class BaseRequest(BaseModel):
    para: str


class BertRequest(BaseRequest):
    pipeline = teacher_pipeline


class DBertRequest(BaseRequest):
    pipeline = student_pipeline


class QBertRequest(BaseRequest):
    pipeline = q_teacher_pipeline


class QDBertRequest(BaseRequest):
    pipeline = q_student_pipeline


app = FastAPI()


@app.post("/classify/bert/")
async def bert_classify(item: BertRequest):
    print(item.pipeline(item.para))
    return {"prediction": item.pipeline(item.para)}


@app.post("/classify/distilled-bert/")
async def distilled_bert_classify(item: DBertRequest):
    return {"prediction": item.pipeline(item.para)}


@app.post("/classify/quantized-bert/")
async def quantized_bert_classify(item: QBertRequest):
    return {"prediction": item.pipeline(item.para)}


@app.post("/classify/quantized-distilled-bert/")
async def bert_classify(item: QDBertRequest):
    return {"prediction": item.pipeline(item.para)}
