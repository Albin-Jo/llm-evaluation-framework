export interface errorMessage {
  message: string;
}

export interface TrackByAction {
  screen: string;
  others?: any;
  actionType: string;
}
export interface TrackingDetails {
  timeStamp: string;
  actionType: string;
  currentRoute: string;
  timeSpent: string | number;
  userAgent: string | undefined;
  windowWidth: string | number;
  windowHeight: string | number;
  screen: string;
  others?: any;
  platform: string | undefined;
  model: string | undefined;
  operatingSystem: any;
  osVersion: any;
  description: string | undefined;
  manufacturer: string | undefined;
}

export interface AlertControl {
  title?: string;
  message: string;
  onAlertClose?: () => void;
  buttonText?: string;
}
export interface ConfirmControl {
  title?: string;
  message: string;
  onAccept?: () => void;
  acceptBtnText?: string;
  onCancel?: () => void;
  cancelBtnText?: string;
}

