using Drumwise.Application.Common.Extensions;
using Drumwise.Application.Common.Interfaces;
using Drumwise.Application.Common.Models.Identity;
using Drumwise.Features.Homeworks;
using MediatR;
using Microsoft.AspNetCore.Mvc;

namespace Drumwise.API.Endpoints;

public static class HomeworkEndpoints
{
    public static void MapHomeworkEndpoints(this IEndpointRouteBuilder endpoints)
    {
        var homeworkGroup = endpoints.MapGroup("/homework").RequireAuthorization();
        
        homeworkGroup.MapPost("/",  async Task<IResult>
            ([FromBody] CreateHomeworkCommand createHomeworkCommand, [FromServices]ISender sender) =>
            {
                var result = await sender.Send(createHomeworkCommand).ConfigureAwait(false);
                
                return result.Item1.Match(
                    onSuccess: result.Item1.ProduceSuccessApiResponse(result.Item2),
                    onFailure: result.Item1.ProduceErrorApiResponse());
            })
            .RequireAuthorization("CanAddHomework")
            .Produces<Guid>();
    }
}