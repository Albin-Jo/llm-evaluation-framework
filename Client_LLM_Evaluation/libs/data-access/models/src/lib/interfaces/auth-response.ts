export interface SecurityToken {
  error: string;
  errorDescription: string;
  access_token: string;
  expires_in: number;
  refresh_expires_in: number;
  refresh_token: string;
  token_type: string;
  session_state: string;
  scope: Date;
  bmtok: string;
  bioMetricToken: string;
  userName: string;
  accessToken: string;
  refreshToken: string;
  isImAlwd: boolean;
}

export interface AccessCode {
  hash: string;
  error: string;
  errorDescription: string;
}
export interface OtpStatusResponse {
  status: number,
  token: SecurityToken
}

export interface loginParameter {
  username: string;
  password: string;
  IsBmTokenRequired: boolean;
}
export interface contactDetails {
  rc: number;
  sms: Sms;
  ivr: Ivr;
  email: Email;
  mobilepush: Mobilepush;
  mobilepushDetails: MobilepushDetails[]
}
export interface Mobilepush {
  channel: number;
  contacts: string[];
}
export interface MobilepushDetails {
  channel: number;
  contacts: string[];
  pushPlatform: number;
}
export interface Email {
  channel: number;
  contacts: string[];
}
export interface Ivr {
  channel: number;
  contacts: string[];
}
export interface Sms {
  channel: number;
  contacts: string[];
}
export interface otpConfiguration {
  tokenInfos: TokenInfos[];
  returnCode: number;
  traceId: string;
}
export interface TokenInfos {
  strSerialNo: string;
  iTkType: number;
  iTkStatus: number;
}
export interface OtpResponse {
  otpConfiguration: otpConfiguration;
  contactDetails: contactDetails;
}
export interface pushResponse {
  pushmsg: string;
  refCode: string;
  returnCode: number;
  traceId: string;
}

export interface pullResponse {
  status: number;
  transStatus: number;
  traceId: string;
  token: SecurityToken
}
export interface tokenRequest {
  refresh_token: string,
  access_token: string

}
export interface DeviceInfoDetail {
  platform: string
  uuid: string
  mode: string
  operatingSystem: string
  osVersion: string
  manufacturer: string
}
