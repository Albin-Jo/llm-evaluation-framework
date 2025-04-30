import { Component, OnInit, } from '@angular/core';

@Component({
  selector: 'app-sub1',
  templateUrl: './sub1.page.html',
  styleUrls: ['./sub1.page.scss']
})
export class Sub1Page implements OnInit {


  async ngOnInit() {
    setTimeout(() => {
      console.log('error');
      throw (new Error('errorror'))
    },
      1000);
  }
}
