from typing import List, Optional, Any
from pydantic import BaseModel

# ================ Question Model ==================

class QuestionModel(BaseModel):
    question_name: str

# ================ ResponsesModel ==================

class ResponseModel(BaseModel):
    value: str
    question: QuestionModel

# ================ Affiliation Model ===============

class AffiliationModel(BaseModel):
    institution: str

# ================ Author Model ====================

class AuthorModel(BaseModel):
    first_name: str
    last_name: str
    orcid_id: Optional[str] = None
    affiliations: List[AffiliationModel]
    presenting: bool

# ================ Title Model ====================

class TitleModel(BaseModel):
    without_html: str

# ================ Submission Model ====================

class SubmissionModel(BaseModel):
    title: List[TitleModel]
    authors: List[AuthorModel]

# ================ Program Session Submission ==========

class ProgramSessionsSubmissionModel(BaseModel):
    submission: SubmissionModel


# ================ Program Column model ================

class ProgramColumnModel(BaseModel):
    name: str
    
# ================ Program Session Column model ======

class ProgramSessionColumnModel(BaseModel):
    program_column: ProgramColumnModel

# ================ Session Model =====================

class SessionModel(BaseModel):
    name: str
    program_sessions_submissions: List[ProgramSessionsSubmissionModel]
    program_sessions_program_columns: List[ProgramSessionColumnModel]
    end_time: str
    start_time: str


# ================ Program Dates Model ===============

class ProgramDatesModel(BaseModel):
    program_sessions: List[SessionModel]
    program_date: str


# ================ Events Model ======================

class EventsByPkModel(BaseModel):
    program_dates: List[ProgramDatesModel]

# ================ Data Model =======================

class DataModel(BaseModel):
    events_by_pk: EventsByPkModel

# ================ RSECon25 Model ===================

class RSECon25Program(BaseModel):
    data: DataModel
