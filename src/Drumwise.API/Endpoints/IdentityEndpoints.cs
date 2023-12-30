using System.Security.Claims;
using Drumwise.Application.Common.Extensions;
using Drumwise.Application.Common.Interfaces;
using Drumwise.Application.Common.Models.Identity;
using Microsoft.AspNetCore.Mvc;

namespace Drumwise.API.Endpoints;

public static class IdentityEndpoints
{
    public static void MapAdditionalIdentityEndpoints(this IEndpointRouteBuilder endpoints)
    {
        var manageGroup = endpoints.MapGroup("/manage").RequireAuthorization();
        
        manageGroup.MapPost("/addAdditionalUserData",  async Task<IResult>
                (ClaimsPrincipal claimsPrincipal, [FromBody] AdditionalUserDataRequest additionalUserDataRequest,
                    [FromServices] IIdentityService identityService) =>
            {
                var result = await identityService
                    .RegisterAdditionalUserData(additionalUserDataRequest, claimsPrincipal);

                return result.Match(
                    onSuccess: result.ProduceSuccessApiResponse(),
                    onFailure: result.ProduceErrorApiResponse());
            })
            .Produces(StatusCodes.Status204NoContent)
            .ProducesValidationProblem()
            .ProducesValidationProblem(StatusCodes.Status404NotFound);
    }
}