import { inject, Injectable } from '@angular/core';
import { Router, NavigationEnd } from '@angular/router';
import * as platform from 'platform';
import { TrackingService } from '@ngtx-apps/data-access/services';
import { TrackByAction, TrackingDetails } from '@ngtx-apps/data-access/models';
import { AnalyticActionTypes, getTimeDiffInSeconds } from '@ngtx-apps/utils/shared';

@Injectable({
  providedIn: 'root',
})
export class AnalyticService {
  private readonly router = inject(Router);
  private readonly trackingService = inject(TrackingService);

  private readonly trackingData: TrackingDetails[];
  constructor() {
    this.trackingData = [];
  }
  init() {
    this.captureRouteChange();
  }

  /**
   * Method to start capturing the route change event and track it
   */
  captureRouteChange() {
    this.router.events.subscribe((event) => {
      if (event instanceof NavigationEnd) {
        const track: TrackByAction = {
          screen: event.url,
          actionType: AnalyticActionTypes.PAGE_ENTER,
        };
        this.saveAudit(track);
      }
    });
  }

  /**
   *
   * Method for calling the API to save the analytics tracking
   */
  async saveAudit(track: TrackByAction) {
    // if (!this.deviceInfo) {
    //   this.deviceInfo = await Device.getInfo();
    // }
    const currentTrackingChanges = this.getCurrentTrackingChanges(track);
    currentTrackingChanges.forEach((track) => {
      this.trackingService.track(track).subscribe(() => {
        console.log('Saved log');
      });
    });
  }

  /**
   *
   * getAnaltics method constructs the Analytical object and returns it
   */
  private getAnalytics(track?: TrackByAction | any): TrackingDetails {
    const _timeStamp = new Date().toISOString();
    let _timeSpentInSeconds = '0'; //in seconds

    if (this.trackingData.length > 0) {
      const prevTimeStamp =
        this.trackingData[this.trackingData.length - 1].timeStamp;
      _timeSpentInSeconds = getTimeDiffInSeconds(_timeStamp, prevTimeStamp);
    }
    const _route =
      track.actionType == AnalyticActionTypes.PAGE_EXIT
        ? track.currentRoute
        : this.router.url;
    const trackingDetails: TrackingDetails = {
      timeStamp: _timeStamp,
      actionType: track.actionType,
      currentRoute: _route,
      timeSpent: _timeSpentInSeconds,
      userAgent: platform.name,
      windowHeight: window.screen.height,
      windowWidth: window.screen.width,
      screen: track.screen,
      others: track?.others ? track.others : '',
      platform: platform.name,
      model: platform.version,
      operatingSystem: platform.os?.family,
      osVersion: platform.os?.version,
      description: platform.description,
      manufacturer: platform.manufacturer,
    };
    return trackingDetails;
  }

  /**
   * Get current tracking changes
   */
  getCurrentTrackingChanges = (track: TrackByAction | any) => {
    const currentTrackingChanges = [];

    if (this.trackingData.length > 0) {
      //Closing the previous tracking
      let prevTrackingDetails = this.trackingData[this.trackingData.length - 1];
      prevTrackingDetails.actionType = AnalyticActionTypes.PAGE_EXIT;
      prevTrackingDetails = this.getAnalytics(prevTrackingDetails);
      currentTrackingChanges.push(prevTrackingDetails);
      this.trackingData.push(prevTrackingDetails);
    }
    //Creating the new tracking
    const currenTrackingDetails: TrackingDetails = this.getAnalytics(track);
    currentTrackingChanges.push(currenTrackingDetails);
    this.trackingData.push(currenTrackingDetails);

    return currentTrackingChanges;
  };
}
