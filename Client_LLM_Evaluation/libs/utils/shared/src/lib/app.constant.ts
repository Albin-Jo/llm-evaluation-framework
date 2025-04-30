import { environment } from './environments/environment';

export const AppConstant = Object.freeze({
  BASE_API_URL: environment.baseApiUrl,
  ENVIRONMENT: environment.environment,
  TIMEOUT: environment.timeOut,
  IDLETIME: environment.idleTime,
  RUNTIME_MODE: environment.runtimeMode,
  KEYCLOCK_AUTHORITY: environment.keyclock_authority,
  KEYCLOCK_REDIRECTURL: environment.keyclock_redirectUrl,
  KEYCLOCK_POSTLOGOUTREDIRECTURI: environment.keyclock_postLogoutRedirectUri,
  KEYCLOCK_CLIENTID: environment.keyclock_clientId,
  KEYCLOCK_SCOPE: environment.keyclock_scope,
  API_URL: environment.apiUrl,
});

/**
 * Constants for the type of action for tracking
 */
export const AnalyticActionTypes = {
  ROUTE_CHANGE: 'ROUTE_CHANGE',
  PAGE_ENTER: 'PAGE_ENTER',
  ROUTE_END: 'ROUTE_END',
  BUTTON_CLICK: 'BUTTON_CLICK',
  PAGE_EXIT: 'PAGE_EXIT',
};

export const AlertConstants = {
  ALERT_TITLE: 'Alert',
  ALERT_BTN_TEXT: 'Okay',
  CONFIRM_TITLE: 'Confirm',
  CONFIRM_ACCEPT_BTN_TEXT: 'Accept',
  CONFIRM_CANCEL_BTN_TEXT: 'Cancel',
};


/**
 * Roles of user
 */
export const ROLES = {
  ALL: 'ALL',
};


export const AuthenticationConstant = {
  biometric_key: 'www.ex.qatarairways.com.qa',
  loading_Message: 'Logging in...',
  otp_validating_message: 'Validating OTP',
  loginResponse_access_token_empty: 'Log in Failed, please try again',
  token: 'token',
  hash: "__ht_hco",
  twoFA_Failed: 'Could not sign you up, please try again',
  twoFA_popup_error: 'Log in Failed ',
  fingerprint_register_message: 'You need to register you Biometric again. Please login using your userid and password to re-registier.',
  fingerprint_Header_message: 'Employee Experience',
  fingerprint_subtitle_message: 'Biometric Authentication,use Fingerprint to Login',
  login_failed: 'login failed,Please verify your credentials',
  pushNotification_failed: 'Push Notification has been failed',
  otp_send_message: 'Sending OTP',
  Please_enter_otp: 'Please enter the OTP',
  enter_valid_OTP: 'Please enter valid OTP',
  twoFactor_autehentication: 'Two Factor Autehentication',
  pushNotification_response: '2FA Authentication is ',
  refreshToken: "refreshToken",
  signoutMessage: "Are you sure you want to signout?",
  isImperAllowed: "__imalwd",
  extendSession: "Your session is going to expire another ",
  runtimeMode_MOBILE: "MOBILE",
  runtimeMode_WEB: "WEB",
  runtimeMode: "runtimeMode"
};
