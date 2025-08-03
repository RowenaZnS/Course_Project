$(document).ready(function () {
    $.get('./media/moviedata/Chinese_Ethnic_Song_and_Dance_Gala/seatmap.json', function (data) {
        console.log(data);
        let drink_data=JSON.parse(window.JSON.stringify(data));
        console.log(data["rows"]);
        //alert(data["rows"][0]["seats"][0].number);
        $(data["rows"]).each(function () {
            console.log(this.rowLabel);
        });
        var innerhtml = `
                <svg data-v-09ae1cf2="" data-v-37eaf5c6="" viewBox="-15 -15 315 165" class="seatplan seatplan">
                    <g>
                        <a data-v-09ae1cf2="">
                            <rect data-v-09ae1cf2="" x="-15" y="-15" rx="2" ry="2" width="315" height="10" style="fill: rgb(0, 0, 0);"></rect> 
                            <text data-v-09ae1cf2="" x="135" y="-10" dominant-baseline="central" text-anchor="middle" style="font-size: 7px; fill: rgb(255,255,255);">
                            銀幕</text>
                        </a>
                    </g> 
            `;
            var y=-15,rownum=-10.5;
        $(data["rows"]).each(function () {
            y+=15;
            rownum+=15;
            var x=0;
            console.log(this.rowLabel);
             innerhtml += `
                <g data-v-09ae1cf2="">
                    <text data-v-09ae1cf2="" x="-7.5" y="${rownum}" width="15" height="15" text-anchor="middle" dominant-baseline="central" style="font-size: 5px; fill: rgb(0,0,0);">
                        ${this.rowLabel}
                    </text> 
                    <text data-v-09ae1cf2="" x="262.5" y="${rownum}" width="15" height="15" text-anchor="middle" dominant-baseline="central" style="font-size: 5px; fill: rgb(0,0,0);">
                        ${this.rowLabel}
                    </text>
            `;
            $(this.seats).each(function () {
                let status=this.status;
                if(status=="available") status="#66ff00";
                else status="#ff3300";
                x+=15;
                //alert(this.number+this.status);
                 innerhtml = innerhtml + `
                    <a>
                        <rect data-v-09ae1cf2="" x="${x}" y="${y}" rx="2" ry="2" width="10" height="10" transform="" style="fill: ${status}; stroke-width: 1; stroke:  ${status};"></rect> <!----> 
                        <text data-v-09ae1cf2="" x="${x+5}" y="${y+4.5}" transform="" dominant-baseline="central" text-anchor="middle" style="font-size: 5px; fill: rgb(0, 0, 0);">
                            ${this.number}
                        </text>
                    </a>`;
            });
            innerhtml = innerhtml +`
                </g>
            `;
        });
        $('#test1').append(innerhtml);
    }).fail(function (error) {
        let innerhtml = `
         <div class="col mb-3">
            <div class="alert alert-danger" role="alert">
                Failed to fetch drink menu. Please try again later.
            </div>
        </div>`;
        $('#menu_container').append(innerhtml);
    });
});
