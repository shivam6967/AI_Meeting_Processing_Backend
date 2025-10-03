from fastapi import APIRouter, UploadFile, File, Depends, Form, BackgroundTasks, Query
from sqlalchemy.orm import Session
from database import get_db
from controller import meetingController
from datetime import datetime

router = APIRouter(
    prefix="/meeting",
    tags=["Meeting"],
    responses={404: {"description": "Not found"}},
)

@router.post("/upload-audio")
async def upload_audio(
    file: UploadFile = File(...),
    session: Session = Depends(get_db),
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    """
    Upload audio file for transcription and processing
    """
    return await meetingController.process_upload(file, session, background_tasks)


@router.get("/processing-status/{task_id}")
def get_processing_status(task_id: str):
    """
    Get the status of a processing task
    """
    return meetingController.get_processing_status(task_id)


@router.get("/get_meetings")
def get_meetings(session: Session = Depends(get_db)):
    """
    Get all meetings
    """
    return meetingController.getMeetings(session)


@router.get("/get_meeting_by_id/{id}")
def get_meeting_by_id(
    id: int,
    language: str = Query("english", description="Language for meeting content: 'english' or 'marathi'"),
    session: Session = Depends(get_db)
):
    """
    Get meeting details by ID with dynamic language translation
    
    Parameters:
    - id: Meeting ID
    - language: Target language ('english' or 'marathi')
    
    Returns meeting data translated to the specified language
    """
    return meetingController.getMeeting(id, session, language)


@router.post("/process_with_language")
async def process_meeting_with_language(
    transcription_id: int,
    language: str = Query("english", description="Processing language: 'english' or 'marathi'"),
    session: Session = Depends(get_db)
):
    """
    Process transcription with specified language
    
    This endpoint allows reprocessing a transcription in a different language
    """
    return meetingController.process_transcirption(transcription_id, session, language)


@router.get("/download_report/{meeting_id}")
async def download_report_endpoint(
    meeting_id: int,
    format: str = Query(..., description="Format type: 'pdf' or 'docx'"),
    language: str = Query("english", description="Report language: 'english' or 'marathi'"),
    db: Session = Depends(get_db)
):
    """
    Download meeting report in specified format (PDF or DOCX) and language
    """
    try:
        print(f"Download request - Meeting ID: {meeting_id}, Format: {format}, Language: {language}")
        return meetingController.download_meeting_report(meeting_id, format, db, language)
    except HTTPException:
        raise
    except Exception as e:
        print(f"Download endpoint error: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Download failed: {str(e)}")


@router.post("/translate_meeting/{meeting_id}")
async def translate_meeting(
    meeting_id: int,
    target_language: str = Query(..., description="Target language: 'english' or 'marathi'"),
    db: Session = Depends(get_db)
):
    """
    Translate an existing meeting to a different language
    
    This is useful for getting a meeting that was originally processed in one language
    translated to another language dynamically
    """
    meeting_data = meetingController.getMeeting(meeting_id, db, "english")
    translated_data = meetingController.translate_meeting_content(meeting_data, target_language)
    return translated_data