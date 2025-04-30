export interface EmployeeDetails {
    userName: string;
    staffNo: string;
    grade: string;
    designation: string;
    doj: Date;
    empId: number;
    staffNO: string;
    firstName: string;
    lastName: string;
    dob: Date;
    gender: string;
    nationality: string;
    martialStatus: string;
    createdBy: string;
    createdDate: Date;
    lastModifiedBy: string;
    lastModifiedDate: Date;
    passportNumber: string;
    passportExpiry: Date;
    qidNumber: string;
    qidExpiry: Date;
    active: string;
    thumbnail: string;
    preferenceId: number;
    pushNotificationFlag: boolean;
    shareProfilePictureFlag: boolean;
    selectedTheme: string;
    TextSize: number;
    defaultHomeScreen: string;
    loginOptions: string;
    notificationChannels: string;
    screenAllignment: string;
    deviceUniqueID: string;
    deviceType: string;
    deviceIMEI: string;
    deviceMAC: string;
    emailAddress: string;
    departmentName: string;
    contactInfo: string;
    srvTenure: string;
}
export interface EmployeePreferences {
    preferenceId: number;
    employeeDetailsId: number;
    pushNotificationFlag: boolean;
    shareProfilePictureFlag: boolean;
    selectedTheme: string;
    textSize: number;
    defaultHomeScreen: string;
    loginOptions: string;
    notificationChannels: string;
    screenAllignment: string;
    deviceUniqueID: string;
    deviceType: string;
    deviceIMEI: string;
    deviceMAC: string;
    isActive: boolean;
    createdBy: string;
    createdDate: Date;
    updatedBy: string;
    updatedDate: Date;
}
export interface PPMDetails {
    EmpOverAllStatus: string;
    pPMFinYear: string;
    empOverAllRating: string;
    empObjectiveKPI: string;
    empCompetency: string;
    mgrOverAllRating: string;
    mgrObjectiveKPI: string;
    mgrCompetency: string;
    staffNo: string;
    empPhotoStream: string;
    mgrPhotoStream: string;
    empOverAllStatus: string;
    staff_Appraisal_Id: number;
    staff_Number: string;
    staff_Seq_Number: number;
    form_Id: number;
    is_Form_Closed: boolean;
    cycle_Id: number;
    staff_MidYear_Competency_Comment: string;
    staff_YearEnd_Competency_Comment: string;
    review_Start_Date: Date;
    review_End_Date: Date;
    manager_Overall_Ratings: string;
    wF_Request_Status: string;
    cycle_Start_Date: Date;
    cycle_End_Date: Date;
    wF_Request_Status_SN: string;
    objective_setting_Date: Date;
    checkIn_Start_Date: Date;
    checkIn_End_Date: Date;
    is_Objective_Approved: boolean;    
    pPM_Phase: string;
    pPM_WF_Request_Status: string;


}
export interface AttendanceDetails {
    ShortageHour: string;
    StaffNo: string;
    OverAllStatus: string;
    ShortageDays: string;
    AttendenceMonth: string;
}

export interface LeaveDetails {
    Entitlment: string;
    Balance: string;
    Utilized: string;
    EntitleFrom: string;
    CarryForward: string;
    EntitleTo: string;
    BalanceId: string;
    CarryForwardSpecial: string;
}
export interface UpdatePreferences {
    preferenceId: number;
    employeeDetailsId: number;
    Preference: string;
    Value: string;
    updatedBy: string;
    updatedDate: Date;
}

